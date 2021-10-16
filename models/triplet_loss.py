import torch
import numpy as np

class TripletLoss(torch.nn.modules.loss._Loss):
    def __init__(self, nb_random_samples, negative_penalty, k_loss_fun='None'):
        super(TripletLoss, self).__init__()
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.k_loss_fun = k_loss_fun

    def forward(self, batch, encoder, train, km=None):
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = train.size(2)

        
# 随机确定pos、neg长短（二者相同），再随机确定anchor长短（>=pos），它们对所有batch相同
# 对于每个sample，分别：随机确定一个anchor区间，再在其中随机选取pos区间
# 对于每个sample，分别：选择k个negative_sample（0<=x<batch_size中抽），再选取选取k个neg区间起始位置


        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = np.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = np.random.randint(1, high=length + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = np.random.randint(
            length_pos_neg, high=length + 1
        )  # Length of anchors
        beginning_batches = np.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = np.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )

        encoder_out = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        ), only_class=False)  # Anchors representations
        representation = encoder_out['repr']
        # K-Means loss
        if self.k_loss_fun == 'contrastive':
            cates = encoder_out['cates'] # gumble softmax choose
            choose = torch.multinomial(cates, 1, replacement=False)
            class_representation = encoder_out['class_repr']
            cores = torch.nn.functional.normalize(encoder.cores, dim=1)
            cates_logits = torch.mm(class_representation, cores.transpose(0,1))
            k_loss = -torch.mean(torch.nn.functional.logsigmoid(cates_logits.gather(1, choose)))
            nb_classes = cates.shape[-1]
            for k in range(nb_classes - 1):
                choose = (choose + 1) % nb_classes
                k_loss += -torch.mean(torch.nn.functional.logsigmoid(-cates_logits.gather(1, choose))) / (nb_classes - 1)
        elif self.k_loss_fun == 'NCE':
            cates = encoder_out['cates']
            class_representation = encoder_out['class_repr'].cpu().detach().numpy()
            pred = torch.from_numpy(km.predict(class_representation)).to(torch.int64).reshape((-1,1)).to(cates.device)
            k_loss = -torch.mean(torch.log(cates.gather(-1, pred)))
        else:
            k_loss = torch.tensor(0)
        
        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ), only_class=False)['repr']  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        
        negative_representations = []
        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)]),
                only_class=False
            )['repr']
            negative_representations.append(negative_representation)
        
        negative_representations = torch.stack(negative_representations, dim=1)
        positive_representation = positive_representation.unsqueeze(1)
        r_reprs = torch.cat([positive_representation, negative_representations], dim=1)
        
        sim = torch.bmm(
            r_reprs,  # B x S x C
            representation.view(batch_size, size_representation, 1)  # B x C x 1
        ).squeeze(2)  # B x S
        
        
        # logistic loss
        loss = -torch.mean(torch.nn.functional.logsigmoid(sim[:, 0])) - torch.mean(torch.nn.functional.logsigmoid(-sim[:, 1:]))
        
        # Xent loss
        # loss = - torch.nn.functional.log_softmax(sim, dim=1)[:, 0].mean()
        
        return loss, k_loss

class TripletLossVaryingLength(torch.nn.modules.loss._Loss):
    def __init__(self, nb_random_samples, negative_penalty, k_loss_fun='None'):
        super(TripletLossVaryingLength, self).__init__()
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.k_loss_fun = k_loss_fun

    def forward(self, batch, encoder, train, km=None):
        batch_size = batch.size(0)
        train_size = train.size(0)
        max_length = train.size(2)

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = np.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Computation of the lengths of the relevant time series
        with torch.no_grad():
            lengths_batch = max_length - torch.sum(
                torch.isnan(batch[:, 0]), 1
            ).data.cpu().numpy()
            lengths_samples = np.empty(
                (self.nb_random_samples, batch_size), dtype=int
            )
            for i in range(self.nb_random_samples):
                lengths_samples[i] = max_length - torch.sum(
                    torch.isnan(train[samples[i], 0]), 1
                ).data.cpu().numpy()

        # Choice of lengths of positive and negative samples
        lengths_pos = np.empty(batch_size, dtype=int)
        lengths_neg = np.empty(
            (self.nb_random_samples, batch_size), dtype=int
        )
        for j in range(batch_size):
            lengths_pos[j] = np.random.randint(
                1, high=lengths_batch[j] + 1
            )
            for i in range(self.nb_random_samples):
                lengths_neg[i, j] = np.random.randint(
                    1,
                    high=lengths_samples[i, j] + 1
                )

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = np.array([np.random.randint(
            lengths_pos[j],
            high=lengths_batch[j] + 1
        ) for j in range(batch_size)])  # Length of anchors
        beginning_batches = np.array([np.random.randint(
            0, high=lengths_batch[j] - random_length[j] + 1
        ) for j in range(batch_size)])  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        # Start of positive samples in the anchors
        beginning_samples_pos = np.array([np.random.randint(
            0, high=random_length[j] - lengths_pos[j] + 1
        ) for j in range(batch_size)])
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + lengths_pos

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.array([[np.random.randint(
            0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
        ) for j in range(batch_size)] for i in range(self.nb_random_samples)])

        out_list = [encoder(
            batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length[j]
            ]
        )for j in range(batch_size)]
        representation = torch.cat([out['repr'] for out in out_list]) # Anchors representations
        # K-Means loss
        if self.k_loss_fun == 'contrastive':
            class_representation = torch.cat([out['class_repr'] for out in out_list])
            cates = torch.cat([out['cates'] for out in out_list])
            choose = torch.multinomial(cates, 1, replacement=False)
            cores = torch.nn.functional.normalize(encoder.cores, dim=1)
            cates_logits = torch.mm(class_representation, cores.transpose(0,1))
            k_loss = -torch.mean(torch.nn.functional.logsigmoid(cates_logits.gather(1, choose)))
        elif self.k_loss_fun == 'NCE':
            class_representation = torch.cat([out['class_repr'] for out in out_list]).cpu().detach().numpy()
            cates = torch.cat([out['cates'] for out in out_list])
            if km != None:
                pred = torch.from_numpy(km.predict(class_representation)).to(torch.int64).reshape((-1,1)).to(cates.device)
            else:
                pred = torch.argmax(cates, dim=-1).reshape((-1,1))
            k_loss = -torch.mean(torch.log(cates.gather(-1, pred)))
        else:
            k_loss = torch.tensor(0)

        positive_representation = torch.cat([encoder(
            batch[
                j: j + 1, :,
                end_positive[j] - lengths_pos[j]: end_positive[j]
            ]
        )['repr'] for j in range(batch_size)])  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations

        negative_representations = []
        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = torch.cat([encoder(
                train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + lengths_neg[i, j]
                ]
            )['repr'] for j in range(batch_size)])
            negative_representations.append(negative_representation)

        negative_representations = torch.stack(negative_representations, dim=1)
        positive_representation = positive_representation.unsqueeze(1)
        r_reprs = torch.cat([positive_representation, negative_representations], dim=1)
        
        sim = torch.bmm(
            r_reprs,  # B x S x C
            representation.view(batch_size, size_representation, 1)  # B x C x 1
        ).squeeze(2)  # B x S
        
        
        # logistic loss
        loss = -torch.mean(torch.nn.functional.logsigmoid(sim[:, 0])) - torch.mean(torch.nn.functional.logsigmoid(-sim[:, 1:]))
        
        # Xent loss
        # loss = - torch.nn.functional.log_softmax(sim, dim=1)[:, 0].mean()

        return loss, k_loss

class TripletLossKm(torch.nn.modules.loss._Loss):
    '''The K-means triplet loss, use only train data from the same cluster as negative sample.'''
    def __init__(self, nb_random_samples, negative_penalty):
        super().__init__()
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train_list, km=None):
        batch_size = batch.size(0)
        max_length = train_list[0].size(2)
        classes = len(train_list)
        train_size_list = []
        for i in range(classes):
            train_size_list.append(train_list[i].size(0))

        # Computation of the lengths of the relevant time series
        with torch.no_grad():
            lengths_batch = max_length - torch.sum(
                torch.isnan(batch[:, 0]), 1
            ).data.cpu().numpy()

        # Choice of lengths of positive and negative samples
        lengths_pos = np.empty(batch_size, dtype=int)
        for j in range(batch_size):
            lengths_pos[j] = np.random.randint(
                1, high=lengths_batch[j] + 1
            )
            
        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = np.array([np.random.randint(
            lengths_pos[j],
            high=lengths_batch[j] + 1
        ) for j in range(batch_size)])  # Length of anchors
        beginning_batches = np.array([np.random.randint(
            0, high=lengths_batch[j] - random_length[j] + 1
        ) for j in range(batch_size)])  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        # Start of positive samples in the anchors
        beginning_samples_pos = np.array([np.random.randint(
            0, high=random_length[j] - lengths_pos[j] + 1
        ) for j in range(batch_size)])
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + lengths_pos

        out_list = [encoder(
            batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length[j]
            ]
        )for j in range(batch_size)]
        representation = torch.cat([out['repr'] for out in out_list]) # Anchors representations
        cates = torch.cat([out['cates'] for out in out_list])
        cates = torch.argmax(cates, dim=1)
        
        positive_representation = torch.cat([encoder(
            batch[
                j: j + 1, :,
                end_positive[j] - lengths_pos[j]: end_positive[j]
            ]
        )['repr'] for j in range(batch_size)])  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations

        # choose negative samples
        samples = np.empty((self.nb_random_samples, batch_size), dtype=int)
        for i in range(batch_size):
            samples[:,i] = np.random.choice(
                train_size_list[cates[i]], size=(self.nb_random_samples)
            )
        samples = torch.LongTensor(samples)

        lengths_samples = np.empty((self.nb_random_samples, batch_size), dtype=int)
        lengths_neg = np.empty((self.nb_random_samples, batch_size), dtype=int)
        with torch.no_grad():
            for i in range(batch_size):
                lengths_samples[:,i] = max_length - torch.sum(
                    torch.isnan(train_list[cates[i]][samples[:,i], 0]), 1
                ).data.cpu().numpy()
                for j in range(self.nb_random_samples):
                    lengths_neg[j, i] = np.random.randint(
                        1,
                        high=lengths_samples[j, i] + 1
                    )

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.array([[np.random.randint(
            0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
        ) for j in range(batch_size)] for i in range(self.nb_random_samples)])

        negative_representations = []
        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(batch_size):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = torch.cat([encoder(
                train_list[cates[i]][samples[j, i]: samples[j, i] + 1][
                    :, :,
                    beginning_samples_neg[j, i]:
                    beginning_samples_neg[j, i] + lengths_neg[j, i]
                ]
            )['repr'] for j in range(self.nb_random_samples)])
            negative_representations.append(negative_representation)

        negative_representations = torch.stack(negative_representations, dim=1).transpose(0,1)
        positive_representation = positive_representation.unsqueeze(1)
        r_reprs = torch.cat([positive_representation, negative_representations], dim=1)
        
        sim = torch.bmm(
            r_reprs,  # B x S x C
            representation.view(batch_size, size_representation, 1)  # B x C x 1
        ).squeeze(2)  # B x S
        
        # logistic loss
        loss = -torch.mean(torch.nn.functional.logsigmoid(sim[:, 0])) - torch.mean(torch.nn.functional.logsigmoid(-sim[:, 1:]))

        return loss
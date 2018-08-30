import torch
import torch.nn as nn
import torch.nn.functional as F


class DEM(nn.Module):
    def __init__(self, cnn, semantic_dim=300, hidden_dim=1024):
        super().__init__()
        self.cnn = cnn
        for p in self.cnn.parameters():
            p.requires_grad = False

        visual_dim = cnn.classifier.in_features

        self.word_emb_transformer = nn.Sequential(*[
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_dim),
            nn.ReLU(),
        ])

    def forward(self, image, label, word_embeddings):
        self.cnn.eval()
        visual_emb, _ = self.cnn(image)
        semantic_emb = self.word_emb_transformer(word_embeddings[label])

        return semantic_emb, visual_emb

    def get_loss(self, image, label, word_embeddings):
        loss = F.mse_loss(*self.forward(image, label, word_embeddings))

        return loss

    def predict(self, image, word_embeddings):
        with torch.no_grad():
            n_class = word_embeddings.size(0)
            batch_size = image.size(0)

            self.cnn.eval()
            visual_emb, _ = self.cnn(image)

            visual_emb = visual_emb.repeat(1, n_class).view(batch_size * n_class, -1)
            semantic_emb = self.word_emb_transformer(word_embeddings).repeat(batch_size, 1)

            score = (visual_emb - semantic_emb).norm(dim=1).view(batch_size, n_class)
            pred = score.min(1)[1]

        return pred.detach().cpu().numpy()
class DeViSE(nn.Module):
    def __init__(self, cnn, hidden_dim=1024, semantic_dim=300):
        super().__init__()
        self.cnn = cnn
         visual_dim = cnn.classifier.in_features
        self.visual_transformer = nn.Sequential(*[
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU()
        ])
        self.word_emb_transformer = nn.Sequential(*[
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU()
        ])
     def forward(self, image, word_embeddings):
        n_class = word_embeddings.size(0)
        batch_size = image.size(0)
         image_feature, logits = self.cnn(image)
         visual_emb = self.visual_transformer(image_feature)
        visual_emb = visual_emb.repeat(1, n_class).view(batch_size * n_class, -1)
         semantic_emb = self.word_emb_transformer(word_embeddings)
        semantic_emb = semantic_emb.repeat(batch_size, 1)
         scores = torch.mul(visual_emb, semantic_emb).sum(dim=1).view(batch_size, -1)
         return logits, scores
     def get_loss(self, image, label, word_embeddings, LAMBDA=0.5):
        logtis, scores = self.forward(image, word_embeddings)
        loss = F.cross_entropy(logtis, label) * LAMBDA + (1 - LAMBDA) * F.multi_margin_loss(scores, label)
         return loss
     def predict(self, image, word_embeddings):
        with torch.no_grad():
            _, score = self.forward(image, word_embeddings)
            pred = score.max(1)[1]
        return pred.detach().cpu().numpy()

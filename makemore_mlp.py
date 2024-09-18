import torch
import torch.nn.functional as F
words = open("names.txt", "r").read().splitlines()

block_size = 3 # context length: how many characters do we take to predict the next char?
# build the vocabulary of characters to/from integers

chars = sorted(set("".join(words)))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

# build the dataset
def build_dataset(words):
  block_size = 3 # context length: how many characters do we take to predict the next char?
  X, Y = [], []

  for w in words:
    # print(w)
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = 0.8*len(words)
n2 = 0.9*len(words)

# train(learn), val(for hyperparameters), test split(for evaluation)
Xtr, Ytr = build_dataset(words[:int(n1)])
Xdev, Ydev = build_dataset(words[int(n1):int(n2)])
Xte, Yte = build_dataset(words[int(n2):])

# creating parameters
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g, requires_grad=True)
W1 = torch.randn((30, 200), generator=g, requires_grad=True)
b1 = torch.randn(200, generator=g, requires_grad=True)
W2 = torch.randn((200, 27), generator=g, requires_grad=True)
b2 = torch.randn(27, generator=g, requires_grad=True)
parameters = [C, W1, b1, W2, b2]

# find a good initial learing rate
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []
epochs = 1000
for i in range(epochs):
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))

  # forward pass
  emb = C[Xtr[ix]]
  h = torch.tanh(emb.view((emb.shape[0], 30)) @ W1 + b1)
  logits = h @ W2 + b2
  # cross entropy
  # counts = logits.exp()
  # probs = counts / counts.sum(1, keepdims=True)
  # loss = -probs[torch.arange(32), Y].log().mean()
  loss = F.cross_entropy(logits, Ytr [ix])
  # print(loss)
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  # update
  # lr = lrs[i] 
  lr = 0.1
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  # lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10(). item())

print(loss)

# dev - evaluate using the dev dataset
emb = C[Xdev]
h = torch.tanh(emb.view((emb.shape[0], 30)) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)

# sampling from the model
for _ in range(10):
  out = []
  context = [0] * block_size
  while True:
    emb = C[torch.tensor([context])]
    h = torch.tanh(emb.view((emb.shape[0], 30)) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, replacement=True)
    context = context[1:] + [ix.item()]
    out.append(ix.item())
    if ix == 0:
      break
  print(''.join(itos[i] for i in out))

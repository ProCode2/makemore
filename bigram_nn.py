import torch

torch.device("cpu")
words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s, i in stoi.items()}

# create a training set of bigrams (x, y)
xs, ys = [], []

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# randomly initialise 27x27 neuron weights, each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)


epochs = 100
for i in range(epochs):
  # forward pass
  import torch.nn.functional as F
  xenc = F.one_hot(xs, num_classes=27).float()
  logits = xenc @ W # we interpret this is log(counts)
  # to get the counts from this we exponentiate the log(counts) or logits 
  counts = logits.exp() # equivalent to N
  # probs[0] is [27] and it says how likely a character is given the input xenc[0] 
  # since all the above ops are differentiable we and backpropagate through them 
  probs = counts / counts.sum(1, keepdims=True)
  # the exponentiation and normalisation is together know as softmax

  # calculate loss (nll for classification)
  # probs[0, ys[0]], probs[1, ys[1]], probs[2, ys[2]], probs[3, ys[3]], probs[4, ys[4]]
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
  print(loss.item())

  # backward pass
  W.grad = None # dont forget the zero_grad, reset the gradients or they will accumulate 
  loss.backward()

  # update
  W.data += -50 * W.grad


# sampling from the neural net model
g = torch.Generator().manual_seed(2147483647)


for i in range(50):
  out = []
  ix = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # we interpret this is log(counts)
    counts = logits.exp() # equivalent to N
    p = counts / counts.sum(1, keepdims=True)


    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break

  print(''.join(out))

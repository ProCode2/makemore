import torch
torch.device("cpu")
words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s, i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)
# smoothening the model
N += 1

# calculating frequencies to find probablity of each bigram
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1

# calculating the probability distributions
P = N.float()
# P.sum(1, keepdim=True) this returns a [27, 1] tensor which is the sum across rows
P /= P.sum(1, keepdim=True) # copies the [27, 1] tensor 27 times and performs element wise divison which normalises the entire array


g = torch.Generator().manual_seed(2147483647)

# generating words using the probablity distribution
for i in range(50):
  out = []
  ix = 0
  while True:
    # p = N[ix].float()
    # p = p / p.sum()
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break

  print(''.join(out))

# calulating the average neg log likelihood of the model
log_likelihood = 0.0
n = 0
# for w in ["praditpq"]:
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    # print(f"{ch1} {ch2}:  {prob:.4f} {logprob:.4f}")


log_likelihood /= n
print(f"{log_likelihood=}")
nll = -log_likelihood
print(f"{nll=}")

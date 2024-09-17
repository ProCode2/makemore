# Text Generation:

# MUltinomial Approach

- A Bigram model is a most naive model for generating characters based on last character. We store bigram frequencies and generate the next character according to that.
- Technical Details: We create a 2D array where rows are characters and columns are characters that follow the characters in rows. we sample from this array, in other words we take a row of start character get it's frequencies, pass it to a torch.multinomial to get a most probable following elem, and we repeat this process until end character is received.
- In this bigram array, the sum across rows and columns is identical (see bigram statistics)
- Learn Broadcasting rules in pytorch, [Link](https://pytorch.org/docs/stable/notes/broadcasting.html)
- The quality of this model is measured by the product of all the probablitis assigned to the bigrams by the model. This is called the likelihood. We try maximise this likelihood, which is equivalent to minimising the neg log likelihood. Since the product will be a very small number, we work with the log of this number, since log(from 0 to 1) is more managable.
- To avoid getting inf neg log likelihood, we make the model more smooth by adding fake counts to all the cells, for ex N += 1 or 5

# Neural Network Approach

- One hot encoding - lets say there are 27 items, to encode 13th item we take a tensor of 27 dimensions, and turn the 13th dimension on(set it to 1), torch.nn.functional.one_hot is a handy function that does that.
- Softmax takes a linear layer emitting count like values and give probability distributions for them. it exponentiates and then normalises the values
- NLL for classification and Mean squared error for regression

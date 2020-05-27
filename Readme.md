# Note:
**Work in progress!** The package has not been finished yet. If you wish to use it, I recommend you check the jupyter notebooks provided first.


![Lyrics Genius](https://github.com/FelipeSulser/Lyrics-Genius/blob/master/assets/LyricsGenius.png)

# Lyrics【Genius】
Lyrics Genius is a Python 3 module to generate text based on a topic and a specific *style*.

With Lyrics Genius you can create a Deep Residual LSTM Network and train it on any text corpus you wish. 

## Features

Lyrics Genius is built on top of Keras and has the following features:

- **Blazingly fast training** 🔥:  Lyrics Genius uses a state of the art neural network architecture that uses skip-connections to improve the quality of the results and accelerate training 

- **Works on any dataset** 📝: Train on any text corpus. Thanks to its character-level encoding neural network, you may train the network on songs or poems of *any language and style or size*. Works great on small datasets too!

- **Autocomplete feature** ✏️: Start writing text and Lyrics Genius will auto-complete the rest with the provided style.

## Architecture

Lyrics Genius' architecture takes inspiration from the [char-rnn](https://github.com/karpathy/char-rnn) network from Andrej Karpathy which I discovered after taking Andrew Ng's [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning).


[**Can we go deeper?**](https://knowyourmeme.com/memes/we-need-to-go-deeper)


A few improvements have been made to char-rnn in order to increase the *style transfer* and the training speed:

- Character embedding model that constructs vectors based on the text characters n-grams.
- 1-Dimensional Spatial Dropout layer to drop entire 1-D feature maps (along all channels).
- [Residual Bidirectional LSTM](https://arxiv.org/abs/1701.03360) units to provide an additional shortcut from the lower layers.


![Network Architecture](https://github.com/FelipeSulser/Lyrics-Genius/blob/master/assets/net_architecture.png)


## Notes

## Next Steps

- Clean API definition
- More examples
- Variable input length (not possible as of now)
- Add Attention architecture to the model

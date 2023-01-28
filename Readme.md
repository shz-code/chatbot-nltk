# Python contextual chatbot with voice recognition.

## Major packages used
- Speech Recognition
- PyTorch
- Pyttsx3

## Project requirements
- `Python 3.7-3.9` (Pytorch limitation)

## Installation Process

First clone this git repository or download zip
```console
  git clone https://github.com/shz-code/chatbot-nltk.git
```
Create a new virtual environment(Use Conda/ Virtual Environment) [Learn More.](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20created,the%20virtual%20environment%20are%20available.)

**If virtual environment is not installed on your machine install it using below command.*
```console
  pip install virtualenv
```
Activate **virtualenv**
```console
   virtualenv env
  .\env\Scripts\activate  
```
Run pip to install all the dependencies
```console
  pip install -r requirements.txt
```
Train your model based on [intents.json](https://github.com/shz-code/chatbot-nltk/blob/master/nlp_pipeline/training%20data/intents.json)

It will create `data.pth` which is the model data.

```console
  py .\nlp_pipeline\train.py
```
If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:

*Run this once in your terminal:*

```console
  python
  >>> import nltk
  >>> nltk.download('punkt')
```
Finally start main.py
```console
  py main.py
```

## Features
- Users can chat or speak with the bot.
- The bot can generate answers based on pre-defined conditions or it will generate answer trained from [intents.json](https://github.com/shz-code/chatbot-nltk/blob/master/nlp_pipeline/training%20data/intents.json) data.
## Customize
You can easily feed more data or customize the model by adding or modifying [intents.json](https://github.com/shz-code/chatbot-nltk/blob/master/nlp_pipeline/training%20data/intents.json) file.

For example
```json
{
  "intents": [
    {
      "tag": "new-tag",
      "patterns": [
        "all the question patterns related to that tag",
        "question 1",
        "question 2"
      ],
      "responses": [
        "all the response patterns related to that tag",
        "response 1",
        "response 2"
      ]
    },
    ...
  ]
}
```
**NB**: You need to train your model again if you have modified the [intents.json](https://github.com/shz-code/chatbot-nltk/blob/master/nlp_pipeline/training%20data/intents.json) file. Run `train.py` file to train your chatbot. If you see no changes happening than delete the old **data.pth** file and train again.



## Acknowledgements
- Main Project idea [Patrick Loeber](https://github.com/patrickloeber/pytorch-chatbot)
- Main article [Contextual Chatbots with TenserFlow]( https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077.) 
- Supporting article [Complete guide to build your ai chatbot with nlp in python](https://www.analyticsvidhya.com/blog/2021/10/complete-guide-to-build-your-ai-chatbot-with-nlp-in-python/)

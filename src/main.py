import torch
import telebot
import config
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt2large")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt2large")
model.to('cpu')

def generate(text):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.to('cpu')
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(
        input_ids, 
        max_length=100 + len(text),
        top_k=0,
        top_p=0.95,
        temperature=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1)
    return tokenizer.decode(greedy_output[0], clean_up_tokenization_spaces=True)

bot = telebot.TeleBot(config.TOKEN)

@bot.message_handler(content_types=["text"])
def repeat_all_messages(message): # Название функции не играет никакой роли, в принципе
    if (message.text == '/author'):
        bot.send_message(message.chat.id, """
Автор бота: Шапошников Игорь
Telegram: @JOURLOY
GitHub: github.com/Jourloy
Написано на Python

Используется модель ruGPT2Large (github.com/sberbank-ai/ru-gpts)
        """)
    elif ('/generate=' in message.text):
        str = message.text.split('=')
        if (len(str[1]) >= 1):
            answer = generate(str[1])
            bot.send_message(message.chat.id, answer)
    elif (message.text == '/settings'):
        bot.send_message(message.chat.id, """
max_length = 100 + len(text)
top_k = 0
top_p = 0.95
temperature = 0.9
repetition_penalty = 1.0
do_sample = True
num_return_sequences = 1
        """)
    elif (message.text == '/help'):
        bot.send_message(message.chat.id, """
/author - Получить информацию об авторе
/settings - Получить информацию о настройках генерации
/generate= - Сгенерировать текст на основе того, который идет сразу после =
Например:
/generate=приветствую вас, Альберт
        """)

if __name__ == '__main__':
    bot.polling(none_stop=True)
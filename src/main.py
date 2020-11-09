import torch
import telebot
import config
from transformers import AutoTokenizer, GPT2LMHeadModel

print('>> Run tokenizer')
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt2large")
print('>> Run model')
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt2large")
print('>> Model to cpu')
model.to('cpu')
print('>> Create bot')
bot = telebot.TeleBot(config.TOKEN)

def generate(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.to('cpu')
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

@bot.message_handler(commands=['start'])
def handle_start(message):
	bot.send_message(message.chat.id, """
Привет! Можешь просто отправить мне текст и я сгенерирую новый на его основе.

Чтобы узнать об авторе, введи или нажми на /author
Чтобы узнать о настройках генерации введи или нажми на /settings
Чтобы узнать о возможных командах введи или нажми на /help
    """)

@bot.message_handler(commands=['help'])
def handle_help(message):
	bot.send_message(message.chat.id, """
/author - Получить информацию об авторе
/settings - Получить информацию о настройках генерации
/generate= - Сгенерировать текст на основе того, который идет сразу после =
Например:
/generate=приветствую вас, Альберт

Также сгенерировать текст можно просто отправив мне обычное сообщение, без всех команд
    """)

@bot.message_handler(commands=['author'])
def handle_author(message):
    bot.send_message(message.chat.id, """
Автор бота: Шапошников Игорь
Telegram: @JOURLOY
GitHub: github.com/Jourloy
Написано на Python

Используется модель ruGPT2Large (github.com/sberbank-ai/ru-gpts)
    """)

@bot.message_handler(commands=['settings'])
def handle_settings(message):
    bot.send_message(message.chat.id, """
max_length = 100 + len(text)
top_k = 0
top_p = 0.95
temperature = 0.9
repetition_penalty = 1.0
do_sample = True
num_return_sequences = 1
    """)

@bot.message_handler(content_types=["text"])
def asnwer(message):
    if ('/generate=' in message.text):
        str = message.text.split('=')
        if (len(str[1]) >= 1):
            print('>> Generate')
            answer = generate(str[1])
            bot.send_message(message.chat.id, answer)
    else:
        print('>> Generate')
        answer = generate(message.text)
        bot.send_message(message.chat.id, answer)

if __name__ == '__main__':
    print('>> Run bot \n')
    bot.polling(none_stop=True)
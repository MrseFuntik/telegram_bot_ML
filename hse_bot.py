import logging
from aiogram import Bot, Dispatcher, types, executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import torch
from pathlib import Path
from PIL import Image
from aiogram.types import InputFile
from fastapi import FastAPI

token = '6593301986:AAEqCYGYQKmDjzssFKimD4rAUdZa_OCLvLA'

logging.basicConfig(level=logging.INFO)
bot = Bot(token=token)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

app = FastAPI()

objects = '''
⏺ small vehicle - маленький автомобиль
⏺ large vehicle - большой автомобиль
⏺ tennis court - теннисный корт
⏺ ground track field - стадион
⏺ ship - корабль
⏺ harbor - гавань (порт)
⏺ storage tank - большой резервуар
⏺ swimming pool - бассейн
⏺ plane - самолет
⏺ bridge - мост
⏺ roundabout - круговой перекрёсток
⏺ baseball diamond - бейсбольное поле
⏺ soccer ball field - футбольное поле
⏺ basketball court - баскетбольная площадка
⏺ helicopter - вертолёт
'''

model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)


@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    await message.answer('Приветствую. Наш бот распознаёт объекты, относящиеся к 15 различным классам, среди которых '
                         'малый и крупный автотранспорт, самолеты, вертолеты, корабли и прочее. '
                         'Отправь мне изображение (к примеру, это может быть снимок со спутника), и я верну тебе '
                         'изображение с обозначенными объектами, если таковые найдутся')


@dp.message_handler(commands=["help"])
async def cmd_help(message: types.Message):
    await message.answer(f"Наш бот классифицирует следующие объекты:\n{objects}\n"
                         "Чтобы получить изображение с классифицируемыми объектами, просто отправь боту изображение. "
                         "Это может быть как сжатое изображение, так и исходный файл, главное чтобы расширение файла "
                         "было либо .png, либо .jpg")


@dp.message_handler(
    content_types=['photo', 'document'])  # от юзера принимает только сжатые фотки / исходники изображений
async def processing_image(message):
    if message.content_type == 'photo':  # сохраняем сжатое изображение
        try:
            await bot.send_message(chat_id=message.chat.id, text='Начинаю обработку изображения...')
            await message.photo[-1].download('satellite_photo.jpg')
            downloaded_image = Path("satellite_photo.jpg")
            results = model(downloaded_image)
            img = results.render()
            img = Image.fromarray(img[0], 'RGB')
            img.save('detection_image.png')
            path = Path("detection_image.png")
            photo = InputFile(path)
            await bot.send_photo(chat_id=message.chat.id, photo=photo)
        except:
            await bot.send_message(chat_id=message.chat.id, text='Что-то пошло не так, попробуйте ещё раз')
    elif message.content_type == 'document':  # сохраняем исходное изображение
        try:  # проверяем, является ли отправленный исходник изображением
            await bot.send_message(chat_id=message.chat.id, text='Начинаю обработку изображения...')
            await message.document.download('satellite_photo.jpg')
            downloaded_image = Path("satellite_photo.jpg")
            results = model(downloaded_image)
            img = results.render()
            img = Image.fromarray(img[0], 'RGB')
            img.save('detection_image.png')
            path = Path("detection_image.png")
            photo = InputFile(path)
            await bot.send_photo(chat_id=message.chat.id, photo=photo)
        except:  # отлавливает ошибку, если попытаться сделать .jpg из исходника, который изображением не является
            await bot.send_message(chat_id=message.chat.id, text='Отправленный файл не является изображением')
            return


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)

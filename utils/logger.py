"""
保存日志信息

"""
import datetime

def save_epoch(epoch):
    with open('../log/log', 'a') as file:
        file.write(f'这是第{epoch}个epoch')

def save_answer(question, answer, epoch, i):
    with open('../log/log', 'a') as file:
        file.write(f'这是第{epoch}个epoch的第{i}条')
        file.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        file.write("question:" + question + '\n' + "answer:" + answer + '\n')
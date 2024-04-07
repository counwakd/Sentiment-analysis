import torch
from torchtext import transforms
from datasets import load_from_disk
from transformers import BertModel , BertTokenizer , AdamW
from transformers.optimization import get_scheduler
from matplotlib import pyplot as plt
import torch.nn as nn

token = BertTokenizer.from_pretrained('bert-base-chinese')

print(token)

#第7章/试编码句子
out = token.batch_encode_plus(
    batch_text_or_text_pairs=['从明天起，做一个幸福的人。', '喂马，劈柴，周游世界。'],
    truncation=True,
    padding='max_length',
    max_length=17,
    return_tensors='pt',
    return_length=True)

#查看编码输出
for k, v in out.items():
    print(k, v.shape)

#把编码还原为句子
print(token.decode(out['input_ids'][0]))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk('./data/ChnSentiCorp')[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label

dataset = Dataset('train')

print(len(dataset), dataset[20])

#第7章/定义计算设备
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(device)

#第7章/数据整理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    #把数据移动到计算设备上
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)

    return input_ids, attention_mask, token_type_ids, labels

#第7章/数据整理函数试算
#模拟一批数据
data = [
    ('你站在桥上看风景', 1),
    ('看风景的人在楼上看你', 0),
    ('明月装饰了你的窗子', 1),
    ('你装饰了别人的梦', 0),
]

#试算
input_ids, attention_mask, token_type_ids, labels = collate_fn(data)

print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

#第7章/数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

print(len(loader))

#第7章/查看数据样例
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break

input_ids.shape, attention_mask.shape, token_type_ids.shape, labels

pretrained = BertModel.from_pretrained('bert-base-chinese')

#统计参数量
sum(i.numel() for i in pretrained.parameters()) / 10000

#第7章/不训练预训练模型,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

#第7章/预训练模型试算
#设定计算设备
pretrained.to(device)

#模型试算
out = pretrained(input_ids=input_ids,
                 attention_mask=attention_mask,
                 token_type_ids=token_type_ids)

print(out.last_hidden_state.shape)

#第7章/定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d (in_channels= 1 , out_channels= 1 , kernel_size= 1)
        self.fc = torch.nn.Linear(in_features=768, out_features=2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        #使用预训练模型抽取数据特征
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids
                             )

        #对抽取的特征只取第一个字的结果做分类即可
        out = torch.as_tensor(out.last_hidden_state[:,0] , dtype=torch.float)
        out = out.reshape(-1 , 1 , 768)
        out = self.conv1(out)
        out = out.reshape(-1 , 768)
        out = self.fc(out)

        out = out.softmax(dim=1)

        return out


model = Model()
print(model)
#设定计算设备
model.to(device)

#试算
model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape

train_loss = []
train_ac = []
train_lr = []

def train():
    #定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-4)

    #定义loss函数
    criterion = torch.nn.CrossEntropyLoss()

    #定义学习率调节器
    scheduler = get_scheduler(name='linear',
                              num_warmup_steps=0,
                              num_training_steps=len(loader),
                              optimizer=optimizer)

    #模型切换到训练模式
    model.train()

    #按批次遍历训练集中的数据
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):

        #模型计算
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

        #计算loss并使用梯度下降法优化模型参数
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        #输出各项数据的情况，便于观察
        if i % 10 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(i, loss.item(), lr, accuracy)
            train_loss.append(loss.item())
            train_lr.append(lr)
            train_ac.append(accuracy)


train()

test_ac = []

#第7章/测试
def test():
    #定义测试数据集加载器
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('test'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    #下游任务模型切换到运行模式
    model.eval()
    correct = 0
    total = 0

    #按批次遍历测试集中的数据
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):
        print(i)

        #计算
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        if i % 10 == 0:
            cc = out.argmax(dim=1)
            accuracy = (cc == labels).sum().item() / len(labels)
            print(i,  accuracy)
            test_ac.append(accuracy)
        #统计正确率
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)

test()


x = range(len(train_loss))
x_label = [i * 10 for i in range(60)]
plt.plot(x_label , train_loss)
plt.title('train_loss')
plt.show()

plt.plot(x_label , train_lr)
plt.title('train_lr')
plt.show()

plt.plot(x_label , train_ac)
plt.title('train_acc')
plt.show()

x_label = [i * 10 for i in range(4)]
plt.plot(x_label , test_ac)
plt.title('test_acc')
plt.show()


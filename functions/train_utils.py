from .data_utils import read_json
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn import metrics


class QwenDataset(Dataset):
    def __init__(self, text, targets, tokenizer, max_len):
        self.max_len = max_len
        self.text = text
        self.tokenizer = tokenizer
        self.targets = targets

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        input_ids = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )

        return {
            'ids': input_ids.input_ids[0],
            'attention_mask': input_ids.attention_mask[0],
            'targets': torch.tensor(self.targets[index], dtype=torch.int64)
        }


def build_dataloader(data_x, data_y, max_len, batch_size, tokenizer):
    train_dataset = QwenDataset(data_x, data_y, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             num_workers=0, shuffle=True, pin_memory=True)
    return train_loader


def train_epoch(epoch, model, train_loader, test_loader, optimizer,
                loss_fn, n_steps_per_epoch, device, lr_scheduler=None):
    model.train()
    epoch_loss = 0
    for step, data in enumerate(train_loader):
        ids = data['ids'].to(device)
        targets = data['targets'].to(device)
        #attention_mask = data['attention_mask'].to(device, dtype=torch.int64)
        outputs = model(ids)
        loss = loss_fn(outputs, targets)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current_lr = optimizer.param_groups[0]['lr']
        if step % 150 == 0:
            print(f'Epoch: {(step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch}, '
                  f'Loss:  {loss.item()}, lr: {current_lr}')

    metrics = {"train/epoch_loss": epoch_loss / len(train_loader),
               "train/learning_rate": optimizer.param_groups[0]['lr']}

    wandb.log(metrics)
    weighted_f1, fin_targets, fin_outputs = evaluate(model,
                                                     test_loader,
                                                     loss_fn,
                                                     device)

    if lr_scheduler is not None:
        lr_scheduler.step()
        # scheduler.step(weighted_f1)
    return weighted_f1, fin_targets, fin_outputs


def test(model, test_iter, loss_fn, device):
    model.eval()
    test_loss, test_F1, fin_targets, fin_outputs = evaluate(model, test_iter, loss_fn, device)
    msg = 'Test Loss: {0:>5.2},  Test F1: {1:>6.2}'
    print(msg.format(test_loss, test_F1))
    print("Precision, Recall and F1-Score...")
    report = metrics.classification_report(fin_targets, fin_outputs, digits=4, zero_division=0)
    print(report)
    return fin_outputs


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    val_loss = 0.
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            ids = data['ids'].to(device)
            targets = data['targets'].to(device)
            # attention_mask = data['attention_mask'].to(device, dtype=torch.int64)
            outputs = model(ids)
            val_loss += loss_fn(outputs, targets) * targets.shape[0]
            outputs = torch.softmax(outputs, dim=-1)
            outputs = torch.argmax(outputs, dim=-1)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

        val_loss = val_loss / len(data_loader.dataset)
        avg_accuracy = round(accuracy_score(fin_targets, fin_outputs) * 100, 2)
        weighted_f1 = f1_score(fin_targets, fin_outputs, average='weighted', zero_division=0)
        micro_f1 = round(f1_score(fin_targets, fin_outputs, average='micro', zero_division=0) * 100, 2)
        macro_f1 = round(f1_score(fin_targets, fin_outputs, average='macro', zero_division=0) * 100, 2)
        val_metrics = {"val/val_loss": val_loss / len(data_loader.dataset),
                       "val/val_accuracy": avg_accuracy,
                       "val/val_weighted_f1": weighted_f1,
                       "val/val_micro_f1": micro_f1,
                       "val/val_macro_f1": macro_f1}
        report = metrics.classification_report(fin_targets, fin_outputs, digits=4, zero_division=0)

        wandb.log(val_metrics)
        '''print("---val_loss:{}, val_accuracy:{}, val_weighted_f1:{}---".format(val_loss / len(data_loader.dataset),
                                                                        avg_accuracy,
                                                                        weighted_f1))'''

    return val_loss, weighted_f1, fin_targets, fin_outputs


def train_val_epoch(epochs, model, train_loader, val_loader, test_loader, optimizer,
                    loss_fn, n_steps_per_epoch, device, lr_scheduler):
    model.train()

    total_batch = 0  # 记录进行到多少batch
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    dev_best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0
        for step, data in enumerate(train_loader):
            ids = data['ids'].to(device)
            targets = data['targets'].to(device)
            # attention_mask = data['attention_mask'].to(device, dtype=torch.int64)
            outputs = model(ids)
            loss = loss_fn(outputs, targets)
            # epoch_loss += loss.item() / targets.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_batch += 1

            if total_batch % 50 == 0:
                true = targets.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_f1 = metrics.f1_score(true, predic, average='weighted', zero_division=0)
                val_loss, val_f1, fin_targets, fin_outputs = evaluate(model,
                                                                      val_loader,
                                                                      loss_fn,
                                                                      device)
                test_loss, test_f1, fin_targets, fin_outputs = evaluate(model,
                                                                        test_loader,
                                                                        loss_fn,
                                                                        device)

                lr_scheduler.step(val_loss.item())

                metrics_wb = {"train/Train_Loss": loss.item(),
                              "train/Train_f1": train_f1,
                              "train/learning_rate": optimizer.param_groups[0]['lr'],
                              "val/Val_Loss": val_loss,
                              "val/Val_f1": val_f1,
                              "test/test_Loss": test_loss,
                              "test/test_f1": test_f1,
                              }

                wandb.log(metrics_wb)

                if val_loss < dev_best_loss:
                    dev_best_loss = val_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''


                msg = 'Iter: {0:>6},  Train Loss: {1:>5.3},  Train f1: {2:>5.3},  Val Loss: {3:>6.4},  Val f1: {4:>5.3},  test f1: {5:>5.3}, learning rate:{6:>6.5}'
                print(msg.format(total_batch, loss.item(), train_f1, val_loss, val_f1, test_f1, optimizer.param_groups[0]['lr']))

            if total_batch - last_improve > 500 or total_batch == 2500:
                fin_outputs = test(model, test_loader, loss_fn, device)
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    # begin to test
    fin_outputs = test(model, test_loader, loss_fn, device)


    return fin_outputs


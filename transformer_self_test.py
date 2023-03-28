from torch import topk
from torch.nn import Transformer
from IPython.display import HTML
import torch


def html_highlight(word, attn):
    html_color = '#%02X%02X%02X' % (
        255, int(255 * (1 - attn)), int(255 * (1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html():
    html = "I am having hard time making the new transformer work. Following code has unexpected(to me) output. "
    html += html_highlight(
        word="Gradients",
        attn=0.5
    )
    html += "<br><br>"
    html += "for the model parameters are zeros and so the optimizer step is of "
    html += html_highlight(
        word="no use",
        attn=0.1
    )
    html += "<br><br>"
    result = HTML(html)
    print(result)


def easy_transformer():
    X = torch.tensor([[[95.0]], [[100.0]], [[105.0]], [[110.0]], [[115.0]]])
    y = torch.tensor([[[120.0]]])
    print(X.shape, y.shape)
    print(X.requires_grad, y.requires_grad)
    model = torch.nn.Transformer(d_model=1, nhead=1, dim_feedforward=100, dropout=0)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    # parms = [j for j in model.parameters()][:3]
    model.train()
    optimizer.zero_grad()
    y_pred = model(X, y)
    print(y_pred)
    print(y)
    print(y_pred.requires_grad)
    print(y_pred._grad)
    loss = criterion(y_pred, y)
    print(loss)
    # for i in parms: print(i._grad)
    loss.backward()
    print(y_pred._grad)
    # for i in parms: print(i._grad)


def transformer_demo():
    zero = torch.zeros((3, 1, 150))  # (批次数，单词长度，向量维度)
    src = torch.fill(zero, 2)
    tgt = torch.fill(zero, 5)
    model = Transformer(d_model=150, nhead=15, num_encoder_layers=12, num_decoder_layers=4, dropout=0.3, norm_first=True)
    out = model(src, tgt)
    print(out)



if __name__ == '__main__':
    # mk_html()
    # easy_transformer()
    transformer_demo()


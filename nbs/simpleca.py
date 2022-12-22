import torch
from torch import nn

class SimpleCA(nn.Module):
    def __init__(self, device, zero_w2=True):
        super().__init__()
        hidden_n=8
        self.filters = torch.stack([torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]]),
          torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]),
          torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]).T,
          torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])]).to(device)
        self.chn = 4
        self.w1 = nn.Conv2d(4*4, hidden_n, 1).to(device)
        self.relu = nn.ReLU()
        self.w2 = nn.Conv2d(hidden_n, 4, 1, bias=False).to(device)
        if zero_w2:
            self.w2.weight.data.zero_()
        self.device = device

    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b*ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:,None])
        return y.reshape(b, -1, h, w)

    def forward(self, x, update_rate=0.5):
        y = self.perchannel_conv(x, self.filters) # Apply the filters
        y = self.w2(self.relu(self.w1(y))) # pass the result through out 'brain'
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w).to(self.device)+update_rate).floor()
        return x+y*update_mask

    def to_rgb(self, x):
        return x[...,:3,:,:]+0.5

    def seed(self, n, sz=128):
        """Initializes n 'grids', size sz. In this case all 0s."""
        return torch.zeros(n, self.chn, sz, sz).to(self.device)

    def to_html(self):
        ws, bias = [p for p in self.w1.parameters()]
        nh = ws.shape[0]
        b1_values = ",".join([str(float(s)) for s in list(bias)])[:-1]
        nh = ws.shape[0]
        w1_values = ''
        for p in ws.squeeze().flatten():
            p = str(float(p))
            w1_values += p
            if not '.' in p:
                w1_values += '.'
            w1_values += ','
        w1_values = w1_values[:-1]
        ws= [p for p in self.w2.parameters()][0]
        w2_values = ''
        for p in ws.squeeze().flatten(): 
            p = str(float(p))
            w2_values += p
            if not '.' in p:
                w2_values += '.'
            w2_values += ','
        w2_values = w2_values[:-1]
        html_text = '\n'.join(open('index.html', 'r').readlines())
        html_text = html_text.replace('B1VALUES', b1_values)
        html_text = html_text.replace('W1VALUES', w1_values)
        html_text = html_text.replace('W2VALUES', w2_values)
        return html_text
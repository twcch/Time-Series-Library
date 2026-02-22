import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Paper Name: DyVol-Fusion (Gated Residual Framework)
    Description: LSTM + Transformer with a Sigmoid Gating Mechanism to predict GARCH residuals.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 從 TSlib 的 configs 接收超參數
        self.seq_len = configs.seq_len       # 輸入的歷史天數 (例如 96)
        self.pred_len = configs.pred_len     # 要預測的未來天數 (例如 24)
        self.enc_in = configs.enc_in         # 輸入的特徵數量 (例如 Open, Close, Volume...共 5 個)
        self.d_model = configs.d_model       # 神經網絡的隱藏層維度 (例如 64 或 128)

        # ---------------------------------------------------
        # 1. AI 引擎 (助教)：負責學習殘差特徵
        # ---------------------------------------------------
        # Step A: LSTM 提取局部趨勢，過濾高頻雜訊
        self.lstm = nn.LSTM(
            input_size=self.enc_in, 
            hidden_size=self.d_model, 
            num_layers=1, 
            batch_first=True
        )
        
        # Step B: Transformer 提取全局長記憶性
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=configs.n_heads, # 注意力頭數
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)

        # 殘差預測頭：把提取出來的特徵，映射到未來的 pred_len 天
        self.residual_head = nn.Linear(self.d_model, self.pred_len)

        # ---------------------------------------------------
        # 2. 動態閘門 (裁判)：決定要不要啟動 AI 路線
        # ---------------------------------------------------
        # 我們拿歷史序列的「最後一天」的市場特徵，來判斷當下有多恐慌
        self.gate_net = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model // 2),
            nn.ReLU(),
            # 輸出一個維度為 pred_len 的向量，代表未來每一天的「啟動權重」
            nn.Linear(self.d_model // 2, self.pred_len), 
            nn.Sigmoid() # 關鍵：壓縮到 0~1 之間
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        TSlib 的標準 forward 函數。
        x_enc shape: [Batch, seq_len, features] (這是你餵進去的歷史數據)
        """
        
        # ==========================================
        # 路徑一：AI 學習真實殘差 (Transformer + LSTM)
        # ==========================================
        # 1. 過 LSTM
        lstm_out, _ = self.lstm(x_enc) # lstm_out shape: [Batch, seq_len, d_model]
        
        # 2. 過 Transformer
        trans_out = self.transformer(lstm_out) # trans_out shape: [Batch, seq_len, d_model]
        
        # 3. 預測未來的殘差 (我們只取最後一個時間點的深層特徵來推論未來)
        # ai_pred shape: [Batch, pred_len]
        ai_pred = self.residual_head(trans_out[:, -1, :]) 

        # ==========================================
        # 路徑二：動態閘門計算權重 (Sigmoid Gate)
        # ==========================================
        # 取歷史輸入的最後一個時間點 x_enc[:, -1, :] 作為市場當下狀態的代表
        # gate_weight shape: [Batch, pred_len]
        gate_weight = self.gate_net(x_enc[:, -1, :]) 

        # ==========================================
        # 最終融合：閘門控制輸出
        # ==========================================
        # 將 AI 的預測值 乘上 閘門的權重 (這就是你說的：學習要不要啟動)
        # 如果 gate_weight 接近 0，輸出就接近 0 (不啟動)
        # 如果 gate_weight 接近 1，輸出就等同 ai_pred (啟動)
        final_residual_out = gate_weight * ai_pred 

        # TSlib 要求輸出格式為 [Batch, pred_len, d_out]
        # 假設我們是預測單一目標 (殘差)，增加最後一個維度
        return final_residual_out.unsqueeze(-1)
import torch.nn as nn
import torch

import numpy as np

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            Riconv_att = False
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        #self.norm = nn.LayerNorm(hidden_size)
        #self.norm2 = nn.LayerNorm(hidden_size)
        self.gamma1 = nn.Parameter(0.1 * torch.ones(hidden_size), requires_grad=True)
        # self.gamma2 = nn.Parameter(1 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = DRIA(hidden_size=hidden_size, num_heads=num_heads,
                             channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate,Riconv_att=Riconv_att)
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1), nn.Conv3d(hidden_size, hidden_size, 3, padding=1, bias=False), nn.GroupNorm(1,hidden_size),)
        self.relu = nn.LeakyReLU()
        self.Riconv_att = Riconv_att
        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        Riconv = None
        if self.Riconv_att:
            Riconv = x[1]
            x = x[0]
            B, C, H, W, D = Riconv.shape
            Riconv = Riconv.reshape(B, C, H * W * D).permute(0, 2, 1)
            #Riconv = torch.nn.functional.normalize(Riconv, dim=-1)
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)


        if self.pos_embed is not None:
            x = x + self.pos_embed
        if Riconv is not None:
            attn = x + self.gamma1 * self.epa_block(x, Riconv)
            # attn = x + self.gamma1 * self.epa_block(x)
        else:
            attn = x + self.gamma1 * self.epa_block(x)

        attn = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn_skip = attn
        #x = self.conv51(attn)
        #attn = self.conv51(attn)
        # attn = self.conv52(attn)
        # x = self.norm(self.conv8(attn_skip + attn))
        x = self.conv8(attn) + attn_skip
        x = self.relu(x)
        return x

class DRIA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, hidden_size, num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1, factor=5, scale=None, Riconv_att = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.factor = factor
        self.scale = scale
        self.Riconv_att = Riconv_att

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
        self.RIC = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        #self.Wr = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # E and F are projection matrices used in spatial attention module to project keys and values from HWD-dimension to P-dimension
        #self.E = nn.Linear(input_size, proj_size)
        #self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj2 = nn.Linear(hidden_size, hidden_size)


    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        # test = torch.matmul(attn, V).type_as(context_in)
        # test2 = context_in[torch.arange(B)[:, None, None],
        #            torch.arange(H)[None, :, None],
        #            index, :]
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)

        return (context_in, None)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)        
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def forward(self, x, Riconv = None):
        B, N, C = x.shape
        if Riconv is not None:
            Riconv = self.RIC(Riconv)
            Riconv = Riconv.reshape(B, N, self.num_heads, C // self.num_heads)
            Riconv = Riconv.permute(0, 2, 1, 3)
        #     #Riconv = Riconv.permute(0, 2, 3, 1)

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        if Riconv is not None:
        #     #q_shared = torch.nn.functional.normalize(q_shared, dim=-2)
        #     #Riconv = torch.nn.functional.normalize(Riconv, dim=-2)
            q_shared = q_shared * Riconv

        q_shared = torch.nn.functional.normalize(q_shared, dim=-2)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-2)

        B, H, L_Q, D = q_shared.shape
        _, _, L_K, _ = k_shared.shape
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        scores_top, index = self._prob_QK(q_shared, k_shared, sample_k=U_part, n_top=u)

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        #v_SA = v_SA.transpose(-2, -1)

        # scale = self.scale or 1. / sqrt(D)
        # if scale is not None:
        #     scores_top = scores_top * scale

        #k_shared_projected = self.E(k_shared)
        #v_SA_projected = self.F(v_SA)

        #k_shared_projected = self.E(k_shared)
        SA_context = self._get_initial_context(v_SA, L_Q)
        SA_context = self.attn_drop_2(SA_context)
        # # update the context with selected top_k queries
        SA_context, attn = self._update_context(SA_context, v_SA, scores_top, index, L_Q)
        # if Riconv is not None:
        #     #Riconv = torch.nn.functional.normalize(Riconv, dim=-1)
        #     q_shared = q_shared * Riconv
        # q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        # k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1))
        attn_CA = self.attn_drop(attn_CA)
        # #
        attn_CA = attn_CA.softmax(dim=-1)
        # #
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)
        # CA_context = self._get_initial_context(v_CA, L_Q)
        # # update the context with selected top_k queries
        # CA_context, attn = self._update_context(CA_context, v_CA, scores_top, index, L_Q)
        x_SA =SA_context.permute(0, 3, 1, 2).reshape(B, N, C)

        #

#        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared) * self.temperature2

#        attn_SA = attn_SA.softmax(dim=-1)
#        attn_SA = self.attn_drop_2(attn_SA)

#        x_SA = (attn_SA @ v_SA.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion

        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        # x = torch.cat((x_SA, x_CA), dim=-1)
        x = x_SA + x_CA
        x = x_CA
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
class Self(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        # E and F are projection matrices used in spatial attention module to project keys and values from HWD-dimension to P-dimension
        # self.E = nn.Linear(input_size, proj_size)
        # self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        # self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_SA = qkvv[0], qkvv[1], qkvv[2]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        # v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        # k_shared_projected = self.E(k_shared)
        #
        # v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        # attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        # attn_CA = attn_CA.softmax(dim=-1)
        # attn_CA = self.attn_drop(attn_CA)

        # x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        # x_CA = self.out_proj2(x_CA)
        # x = torch.cat((x_SA, x_CA), dim=-1)
        return x_SA
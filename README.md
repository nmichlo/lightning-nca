<p align="center">
    <h1 align="center">🦠️ Neural Cellular Automata ⚡️</h1>
    <h3 align="center">Built with PyTorch Lightning</h3>
</p>

<br/>

<p align="center">
    <img align="center" src="resources/docs/in.png"/>
    <img align="center" src="resources/docs/out_col64_comp128.gif"/>
</p>

<br/>

## 🔬 &nbsp;Why?

- PyTorch Lightning implementations are clean
- I wanted to learn about these for fun

## 🧪 &nbsp;Get Started

1. Clone this repository

2. Install the requirements `pip3 install -r requirements.txt`

3. Run the initial example in `python3.9 nca/example_texture_nca.py` which is a port of the
   simple pytorch example for [Self-Organising Textures](https://distill.pub/selforg/2021/textures/)

   ```bash
   python3.9 nca/example_texture_nca.py --train_steps=500 run --lr=0.0005
   #                                                       /
   #  'run' separates trainer args from NCA args ---------/
   ```
   
   See all available arguments:

   - **General Training Arguments**
 
     ```bash
     python3.9 nca/example_texture_nca.py --help
     ```
 
     ```
     --train_steps=TRAIN_STEPS
         Type: int
         Default: 5000
     --train_cuda=TRAIN_CUDA
         Type: bool
         Default: False
     --vis_period_plt=VIS_PERIOD_PLT
         Type: int
         Default: 500
     --vis_period_vid=VIS_PERIOD_VID
         Type: int
         Default: 2500
     --vis_im_size=VIS_IM_SIZE
         Type: int
         Default: 256
     --vis_out_dir=VIS_OUT_DIR
         Type: str
         Default: 'out/2021-06-11_22-09-29'
     --plt_show=PLT_SHOW
         Type: bool
         Default: False
     ```
 
   - **NCA Arguments**
 
     ```
     python3.9 nca/example_texture_nca.py run --help
     ```
 
     ```
     --style_img_uri=STYLE_IMG_URI
         Type: str
         Default: 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/T...
     --style_img_size=STYLE_IMG_SIZE
         Type: int
         Default: 128
     --nca_img_size=NCA_IMG_SIZE
         Type: int
         Default: 128
     --nca_learn_filters=NCA_LEARN_FILTERS
         Type: bool
         Default: False
     --nca_channels=NCA_CHANNELS
         Type: int
         Default: 12
     --nca_hidden_channels=NCA_HIDDEN_CHANNELS
         Type: int
         Default: 96
     --nca_pad_mode=NCA_PAD_MODE
         Type: str
         Default: 'circular'
     --nca_default_update_ratio=NCA_DEFAULT_UPDATE_RATIO
         Type: float
         Default: 0.5
     --iters=ITERS
         Type: typing.Tuple[int, int]
         Default: (32, 64)
     --batch_size=BATCH_SIZE
         Type: int
         Default: 4
     --lr=LR
         Type: float
         Default: 0.001
     --pool_size=POOL_SIZE
         Type: int
         Default: 1024
     --pool_reset_element_period=POOL_RESET_ELEMENT_PERIOD
         Type: int
         Default: 2
     --pool_on_cpu=POOL_ON_CPU
         Type: bool
         Default: True
     --normalize_gradient=NORMALIZE_GRADIENT
         Type: bool
         Default: True
     --consistency_loss_scale=CONSISTENCY_LOSS_SCALE
         Type: Optional[float]
         Default: None
     --scale_loss=SCALE_LOSS
         Type: Optional[float]
         Default: None
     ```

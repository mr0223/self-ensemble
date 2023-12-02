# Prepare the following based on DIRECT corpus (https://github.com/junya-takayama/DIRECT/)
# 1. input/{train/valid/test}.{direct/indirect/w.direct/w.indirect}
#    - Stores input text to BART ("w" includes dialogue history)
# 2. bart/{I2D/D2I/I2D_w/D2I_w}_bart
#    - Stores BART models trained on the DIRECT corpus (I2D corresponds to indirect-to-direct task)

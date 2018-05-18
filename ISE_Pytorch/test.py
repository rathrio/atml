import train
import time

data = 'f30k'
saveto = 'vse/%s' %data
encoder = 'lstm'

if __name__ == "__main__":
    begin_time = time.time()
    train.trainer(data=data, dim_image=4096, lrate=1e-3, margin=0.2, encoder=encoder, max_epochs=100, batch_size=16,
                dim=1000, dim_word=300, maxlen_w=150, dispFreq=10, validFreq=100, early_stop=40, saveto=saveto)

    print('Using %.2f s' %(time.time()-begin_time))

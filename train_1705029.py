
import glob
import cv2
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
from sklearn import metrics
def preprocess(set_X):
    X=255-set_X
    X=np.where(X<80,0,255)
    X=X/255
    # mean=np.mean(X)
    # std=np.std(X)
    # X=(X-mean)/std
    return X
def load_numtaDB_dataset():
    cv_img=glob.glob("training-a/*.png")
    #cv_img2=cv_img+(glob.glob("training-b/*.png"))
    #cv_img3=cv_img2+(glob.glob("training-c/*.png"))
    # for img in glob.glob("training-a/*.png"):
    #     n = cv2.imread(img)
    #     cv_img.append(n)
    # for img in glob.glob():
    #     n = cv2.imread(img)
    #     cv_img.append(n)
    # for img in glob.glob():
    #     n = cv2.imread(img)
    #     cv_img.append(n)
    training_validation_set_X=np.array([cv2.resize(cv2.imread(f),(32,32),interpolation=cv2.INTER_LINEAR) for f in cv_img])
    training_validation_set_X=preprocess(training_validation_set_X)
    # training_validation_set_X=255-training_validation_set_X
    # training_validation_set_X=np.where(training_validation_set_X<100,0,255)
    # training_validation_set_X=training_validation_set_X/255
    #
    #
    # print(training_validation_set_X.shape)
    df=pd.read_csv("training-a.csv",usecols=['digit'])
    training_validation_set_Y=df.to_numpy()
    training_validation_set_Y=training_validation_set_Y.reshape(-1)




    # print(training_validation_set_Y.shape)
    # print(test_set_Y.shape)

    validation_set_size=int(training_validation_set_Y.shape[0]/10)
    train_size=training_validation_set_Y.shape[0]-validation_set_size
    # print(validation_set_size)
    # print(train_size)

    no_of_data_points=training_validation_set_Y.shape[0]
    arr=np.arange(no_of_data_points)
    train_idxs=arr[:train_size]
    validation_idxs=arr[train_size:]
    training_set_X=training_validation_set_X[train_idxs,:]
    training_set_Y=training_validation_set_Y[train_idxs]
    validation_set_X=training_validation_set_X[validation_idxs,:]
    validation_set_Y=training_validation_set_Y[validation_idxs]
    #training_Y=training_validation_set_Y[:,train_size]
    # print(training_Y.shape)
    print(training_set_X.shape)
    print(validation_set_X.shape)

    # print("hi")
    return training_set_X,training_set_Y,validation_set_X,validation_set_Y



class Convolution:
    def __init__(self,no_of_output_channels,filter_dim,stride,padding,input_channels):
        self.number_of_channels=input_channels
        self.filter_dim=filter_dim
        self.padding=padding
        self.no_of_kernels=no_of_output_channels
        self.stride_h=stride[0]
        self.stride_w=stride[1]
        conv_in=self.filter_dim*self.filter_dim*input_channels
        initialize_denom=np.sqrt(2/conv_in)
        self.kernel=np.random.randn(self.filter_dim,self.filter_dim,input_channels,no_of_output_channels)*initialize_denom
        temp_bias=np.zeros(self.no_of_kernels)
        self.bias=temp_bias.reshape(1,1,1,self.no_of_kernels)
    def convolution(self,input,kernel,strided_shape,strided_strides,no_of_ins):
        if no_of_ins==6:
            expanded=np.lib.stride_tricks.as_strided(input,(strided_shape[0],strided_shape[1],strided_shape[2],strided_shape[3],strided_shape[4],strided_shape[5]),strided_strides,writeable=False)
            sum=np.einsum('yzwo,yzwtcm->tcmo',kernel,expanded)
        else:
            expanded=np.lib.stride_tricks.as_strided(input,(strided_shape[0],strided_shape[1],strided_shape[2],strided_shape[3],strided_shape[4],strided_shape[5],strided_shape[6]),strided_strides,writeable=False)
            sum=np.einsum('yzwo,yzwtcmk->tcmo',kernel,expanded)
        return sum
    def forward_pass(self,inputs):
        # print("hereee")
        # print(inputs.shape)
        # print(self.padding)
        inputs=np.pad(inputs,((0,),(self.padding,),(self.padding,),(0,)))
        self.prev_input=inputs
        # print("tgere")
        dim_1=inputs.shape[1]-self.kernel.shape[0]
        kern_width=self.filter_dim
        dim_2=inputs.shape[2]-kern_width
        outputheight=int(np.floor(dim_1+self.stride_h/self.stride_h))
        outputwidth=int(np.floor(dim_2+self.stride_w/self.stride_w))
        batch_str,height_str,width_str,channel_str=inputs.strides
        out_h_str=height_str*self.stride_h
        out_w_str=width_str*self.stride_w
        strided_shapes=[self.filter_dim,self.filter_dim,self.kernel.shape[2],inputs.shape[0],outputheight,outputwidth]
        strides=[height_str,width_str,channel_str,batch_str,out_h_str,out_w_str]
        conv=self.convolution(inputs,self.kernel,strided_shapes,strides,6)
        result=conv+self.bias
        return result

    def backward_pass(self,gradiant,learning_rate):
        div=gradiant.shape[0]
        temp_db=np.sum(gradiant,axis=(0,1,2))/div
        db=temp_db.reshape(self.bias.shape)
        temp_1=(gradiant.shape[1]-1)*self.stride_h
        temp_1+=1
        temp_2=(gradiant.shape[1]-1)*self.stride_w
        temp_2+=1
        input_dim_pad_1=int(temp_1)
        input_dim_pad_2=int(temp_2)
        gradiant_dialated=np.zeros((gradiant.shape[0],input_dim_pad_1,input_dim_pad_2,self.no_of_kernels))
        gradiant_dialated[:,::self.stride_h,::self.stride_w,:]=gradiant
        inputs=self.prev_input
        kernel=gradiant_dialated
        dim_1=inputs.shape[1]-kernel.shape[1]
        dim_2=inputs.shape[2]-kernel.shape[2]
        outputheight=int(np.floor(dim_1+1))
        outputwidth=int(np.floor(dim_2+1))
        extended_strides=2*inputs.strides
        extended_strides=extended_strides[1:]
        expanded = np.lib.stride_tricks.as_strided(inputs,(outputheight,outputwidth,gradiant.shape[3],inputs.shape[3],gradiant.shape[1],gradiant.shape[2],1),
        (extended_strides),writeable=False)
        dw= np.einsum('xklw,pqwtkly->pqtw',kernel,expanded)
        temp_dialated=gradiant_dialated
        dim_to_pad=self.kernel.shape[0]-self.padding
        dim_to_pad=dim_to_pad-1
        gradaiant_dialated_pad=np.pad(temp_dialated,((0,),(dim_to_pad,),(dim_to_pad,),(0,)))

        # flipp_filter=self.kernel[::-1,::-1,:,:]
        transposed_filter=np.transpose(self.kernel,(0,1,3,2))
        flipp_filter=np.rot90(transposed_filter,1,axes=(0,1))
        flipp_filter=np.rot90(flipp_filter,1,axes=(0,1))
        inputs=gradaiant_dialated_pad
        kernel=flipp_filter
        dim_1=inputs.shape[1]-kernel.shape[0]
        dim_2=inputs.shape[2]-kernel.shape[1]
        outputheight=int(np.floor(dim_1+1))
        outputwidth=int(np.floor(dim_2+1))
        extended_strides=2*inputs.strides
        extended_strides=extended_strides[1:]
        self.kernel=self.kernel-learning_rate*dw
        strided_shapes=[flipp_filter.shape[0],flipp_filter.shape[1],flipp_filter.shape[2],inputs.shape[0],outputheight,outputwidth,1]
        strides=extended_strides
        dx=self.convolution(inputs,kernel,strided_shapes,strides,7)
        self.bias=self.bias-learning_rate*db
        self.prev_input=None
        return dx

    def memory_clear(self):
        self.prev_input = None
class Relu:
    def forward_pass(self,inputs):
        self.prev_input=inputs
        # print("hereee")
        output=inputs
        output[output<=0]=0
        return  output
    def backward_pass(self,gradiant,learning_rate):
        derivative=np.where(self.prev_input<=0,0,1)
        dr=gradiant*derivative
        self.prev_input=None
        return dr
    def memory_clear(self):
        self.prev_input = None
class MaxPool:
    def __init__(self,pool_size,stride):
        self.pool_size=pool_size
        self.stride=stride
        self.axises=[1,2]
    def forward_pass(self,inputs):
        self.prev_input=inputs
        sample=inputs.shape[0]
        h=inputs.shape[1]
        w=inputs.shape[2]
        channel=inputs.shape[3]
        out_h=int(np.floor((h-self.pool_size)/self.stride))
        out_h=out_h+1
        out_w=int(np.floor((w-self.pool_size)/self.stride))
        out_w=out_w+1
        batch_str=None
        height_str=None
        width_str=None
        channel_str=None
        batch_str,height_str,width_str,channel_str=inputs.strides
        strides=[batch_str,height_str*self.stride,width_str*self.stride,channel_str,height_str,width_str]
        expand=np.lib.stride_tricks.as_strided(inputs,
                                               (sample,out_h,out_w,channel,self.pool_size,self.pool_size),
                                               (strides))
        dim_1=out_h*self.stride
        dim_2=out_w*self.stride
        output=np.max(expand,axis=(4,5))
        temp=output
        for i in self.axises:
            maxs=temp.repeat(self.pool_size,axis=i)
            temp=maxs
        window = np.empty((inputs.shape[0], dim_1, dim_2, inputs.shape[-1]))
        window=inputs[:,:dim_1,:dim_2,:]
        #window = inputs[np.arange(inputs.shape[0])[:, np.newaxis, np.newaxis], :dim_1, :dim_2, np.newaxis]
        # mask=np.zeros_like(window, dtype=int)
        # max_value=np.max(window,axis=(1,2))
        # mask[window==max_value]=1
        mask = np.where(np.equal(window,maxs),1,0)
        self.mask=mask
        return output
    def backward_pass(self,gradiant,learning_rate):
         derivative=None
         for i in self.axises:
             derivative=gradiant.repeat(self.pool_size,axis=i)
             gradiant=derivative

         da=derivative*self.mask
         #pad initialize
         padded = np.zeros_like(self.prev_input)
         dims=da.shape
         padded[:,:dims[1],:dims[2],:]=da
         result=padded
         self.prev_input=None
         self.mask=None
         return result

    def memory_clear(self):
        self.prev_input = None
        self.mask = None

class Flattening:
    def forward_pass(self,inputs):
        self.prev_input=inputs
        # print(inputs.shape)
        # print(inputs.shape[1])
        second_dim=inputs.shape[1]
        for iter in range((len(inputs.shape)-2)):
            second_dim=second_dim*inputs.shape[2+iter]
        second_dim=np.prod(inputs.shape[1:])
        inputs=inputs.reshape(inputs.shape[0],second_dim)
        return inputs
    def backward_pass(self,gradiant,learning_rate):
        gradiant=np.reshape(gradiant,self.prev_input.shape)
        self.prev_input=None
        return gradiant

    def memory_clear(self):
        self.prev_input = None
class Fully_Connected:
    def __init__(self,no_of_output,no_of_input):
        self.input_dim=no_of_input
        self.output_dim=no_of_output
        xavier_denom=np.sqrt(2/self.input_dim)
        self.weights=np.random.randn(self.output_dim,self.input_dim)*xavier_denom
        self.bias=np.zeros((1,self.output_dim))
        self.bias=self.bias.T
        self.flat=Flattening()

    def forward_pass(self,previous_image):
        flat_image=self.flat.forward_pass(previous_image)
        self.prev_input=flat_image
        output=self.weights @ flat_image.T
        output=output+self.bias
        return output.T
    def update(self,dw,db,learning_rate):
        grad_w=learning_rate*dw
        grad_bias=learning_rate*db
        self.weights=self.weights-grad_w
        self.bias=self.bias-grad_bias
    def backward_pass(self,gradiant,learning_rate):
        dw=np.transpose(gradiant) @ self.prev_input
        dx=gradiant @ self.weights
        db=np.sum(gradiant.T,axis=1,keepdims=True)
        self.update(dw,db,learning_rate)
        #fl=Flattening_layer()
        self.prev_input=None
        return self.flat.backward_pass(dx,learning_rate)

    def memory_clear(self):
        self.prev_input = None

class Softmax:
    def forward_pass(self,inputs):
        trans_in=inputs.T
        trans_in=trans_in-np.max(trans_in,axis=0,keepdims=True)
        v=np.exp(trans_in)
        v=v/np.sum(v,axis=0)
        return v
    def backward_pass(self,gradiant,learning_rate):
        return gradiant

    def memory_clear(self):
        pass

def loss_func(y_pred,y_true):
    temp=-np.multiply(np.log(y_pred), y_true)
    result=temp.sum()
    return result
def batch_normalization(batch_X,beta,gamma,epsion=1e-6):
    mean=np.mean(batch_X,axis=(1,2))
    var=np.var(batch_X,axis=(1,2))
    eps=var+epsion
    norm=(batch_X-mean)/np.sqrt(eps)
    return beta+(gamma*norm)

def generate_one_hot(shape,value,flag):
    one_hot_encoding=np.zeros(shape)
    if flag=="Transpose":
        one_hot_encoding[np.arange(value.shape[0]),value] = 1
    else:
        one_hot_encoding[value,np.arange(value.shape[0])] = 1
    return one_hot_encoding

class Lenet_Model:
    def __init__(self,training_X,training_Y,validation_X,validation_Y,lr,batch_size):
        full_1_neuron=120
        full_2_neuron=84
        full_3_neuron=10
        fc_out=400
        self.train_X=training_X
        self.train_Y=training_Y
        self.layers=[]
        stride_array=np.array([1,1])
        conv1=Convolution(6,5,stride_array,0,self.train_X.shape[3])
        self.layers.append(conv1)
        last_no_of_kernels=6
        relu1=Relu()
        self.layers.append(relu1)
        max_pool_1=MaxPool(2,2)
        self.layers.append(max_pool_1)
        conv2=Convolution(16,5,stride_array,0,last_no_of_kernels)
        self.layers.append(conv2)
        max_pool_2=MaxPool(2,2)
        relu_2=Relu()
        self.layers.append(relu_2)
        self.layers.append(max_pool_2)
        full_1=Fully_Connected(full_1_neuron,fc_out)
        self.layers.append(full_1)
        full_2=Fully_Connected(full_2_neuron,full_1_neuron)
        self.layers.append(full_2)
        full_3=Fully_Connected(full_3_neuron,full_2_neuron)
        self.layers.append(full_3)
        soft=Softmax()
        self.layers.append(soft)
        self.validation_X=validation_X
        self.validation_Y=validation_Y
        self.learning_rate=lr
        self.batch_size=batch_size
    def forward_pass(self,batch_X,batch_Y):
        X=batch_X
        for i in range (len(self.layers)):
            X=self.layers[i].forward_pass(X)
        y_pred=X

        # print("hereeeeeeeeeeee")
        # print(pred_label)
        one_hot_encoding = generate_one_hot(y_pred.shape,batch_Y,"No")
        pred_label=np.argmax(y_pred,axis=0)
        acc=accuracy_score(batch_Y,pred_label)
        loss=loss_func(y_pred,one_hot_encoding)/batch_X.shape[0]
        return y_pred.T,acc,loss
    def forward_pass_without_label(self,batch_X):
        X=batch_X
        for i in range (len(self.layers)):
            X=self.layers[i].forward_pass(X)
        y_pred=X
        return y_pred.T
    def backward_pass(self,gradiant):
        X=gradiant
        for i in self.layers[::-1]:
            X=i.backward_pass(X,self.learning_rate)

    def mem_clear(self):
        for i in range(len(self.layers)):
            self.layers[i].memory_clear()
        self.train_X = None
        self.train_Y = None
        self.validation_X = None
        self.validation_Y = None

    def validate(self):
        predicted_Y, acc, loss = self.forward_pass(self.validation_X, self.validation_Y)
        acc = acc * 100.0
        pred_label = np.argmax(predicted_Y.T, axis=0)
        f_score = f1_score(self.validation_Y, pred_label, average='macro')
        print("Validation loss:", loss, "\nValidation accuracy:", acc, "%\nmacro_F1 score:", f_score)
        return f_score, acc, loss
    def test(self,test_X,test_Y,flag):
        if flag==False:
            predicted_Y=self.forward_pass_without_label(test_X)
            pred_label = np.argmax(predicted_Y.T, axis=0)
        else :
            predicted_Y,acc,loss=self.forward_pass(test_X,test_Y)
            pred_label = np.argmax(predicted_Y.T, axis=0)
            acc=acc*100.0
            f_score=f1_score(test_Y,pred_label, average='macro')
            print("Test loss:", loss, "\nTest accuracy:", acc, "%\nmacro_F1 score: ", f_score)
            labels = np.arange(0, 10).tolist()
            confusion_matrix = metrics.confusion_matrix(test_Y, pred_label)
            dis = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
            dis.plot()
            plt.show()
        return pred_label

    def train(self):
        prev_acc = 0
        epoch = 0
        no_of_samples = self.train_X.shape[0]
        train_acc = []
        train_loss = []
        train_f1 = []
        val_acc = []
        val_loss = []
        val_f = []

        while True:
            accumulated_acc = 0
            accumulated_loss = 0
            accumulated_f1 = 0
            no_of_runs = int(no_of_samples / self.batch_size)
            print("Epoch: ", epoch + 1)
            for i in tqdm(range(no_of_runs), desc="Progress per Epoch"):
                idx_start = i * self.batch_size
                idx_end = idx_start + self.batch_size
                sub_batch_X = self.train_X[idx_start:idx_end]
                sub_batch_Y = self.train_Y[idx_start:idx_end]
                predicted_Y, Acc, lo = self.forward_pass(sub_batch_X, sub_batch_Y)
                pred_label = np.argmax(predicted_Y.T, axis=0)
                f_score = f1_score(sub_batch_Y, pred_label, average='macro')
                one_hot = generate_one_hot(predicted_Y.shape, sub_batch_Y, "Transpose")
                gradiant = (predicted_Y - one_hot) / self.batch_size
                gradiant = self.backward_pass(gradiant)
                accumulated_acc = accumulated_acc + Acc
                accumulated_loss = accumulated_loss + lo
                accumulated_f1 = accumulated_f1 + f_score

            accumulated_loss = accumulated_loss / no_of_runs
            train_loss.append(accumulated_loss)
            accumulated_f1 = accumulated_f1 / no_of_runs
            train_f1.append(accumulated_f1)
            accumulated_acc = (accumulated_acc / no_of_runs) * 100
            train_acc.append(accumulated_acc)

            print("Train loss:", accumulated_loss, "\nTraining accuracy:", accumulated_acc, "%")
            # if (prev_acc-accumulated_acc)>=5:
            #     break

            prev_acc = accumulated_acc
            epoch = epoch + 1
            val_f1, val_ac, val_lo = self.validate()
            val_f.append(val_f1)
            val_loss.append(val_lo)
            val_acc.append(val_ac)
            if (epoch == 15):
                break
        print("train accc")
        print(train_acc)
        print("epoch")
        epoch_list = np.arange(1, epoch + 1).tolist()
        print(epoch_list)
        print("==========Final Result=============")
        print("Train accuracy:", accumulated_acc)
        print("Train loss:", accumulated_loss)
        print("Train F1:", accumulated_f1)
        print("Validation accuracy:", val_ac)
        print("Validation loss:", val_lo)
        print("Validation F1:", val_f1)

        plt.plot(epoch_list, train_acc, color='red', label='training accuracy')
        plt.plot(epoch_list, val_acc, color='blue', label='validation accuracy')
        plt.xlabel("Epoch count")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        plt.plot(epoch_list, train_loss, color='red', label='training loss')
        plt.plot(epoch_list, val_loss, color='blue', label='validation loss')
        plt.xlabel("Epoch count")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.plot(epoch_list, train_f1, color='red', label='training f1')
        plt.plot(epoch_list, val_f, color='blue', label='validation f1')
        plt.xlabel("Epoch count")
        plt.ylabel("Macro_F1 score")
        plt.legend()
        plt.show()
            # if epoch==3:
            #     break

if __name__ == '__main__':
    training_X, training_Y, validation_X, validation_Y = load_numtaDB_dataset()

    # ekhane test_x,test_y dite hobeeeeeeeee
    Model = Lenet_Model(training_X, training_Y, validation_X, validation_Y, 0.01, 32)
    Model.train()

    print("train hoiseeeeeee")
    Model.mem_clear()
    file_name="1705029_model.pickle"
    with open(file_name, 'wb') as file:
        pickle.dump(Model,file)
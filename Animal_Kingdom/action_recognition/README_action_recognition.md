# Action Recognition

## Dataset and Code
* [Download dataset and code here](https://forms.office.com/r/WCtC0FRWpA)

## Structure of Action Recognition Dataset
* Annotations follow Charades format and are stored in .csv format 
* Annotations:
    * `clip_id`
    * `clip_number`
    * `frame_number`
    * `clip_path`
    * `action_labels` 

## Evaluation Metric
* We follow Charades and use mAP for multi-label outputs.
* For the evaluation code, please refer to <https://github.com/facebookresearch/SlowFast/blob/2090f2918ac1ce890fdacd8fda2e590a46d5c734/slowfast/utils/meters.py#L231>

* For reference:
    <details><summary>Click to see code</summary>

    ```python script
    def get_map(preds, labels):
        """
        Compute mAP for multi-label case.
        Args:
            preds (numpy tensor): num_examples x num_classes.
            labels (numpy tensor): num_examples x num_classes.
        Returns:
            mean_ap (int): final mAP score.
        https://github.com/facebookresearch/SlowFast/blob/2090f2918ac1ce890fdacd8fda2e590a46d5c734/slowfast/utils/meters.py#L231
        """
        preds = preds[:, ~(np.all(labels == 0, axis=0))]
        labels = labels[:, ~(np.all(labels == 0, axis=0))]
        aps = [0]
        try:
            aps = average_precision_score(labels, preds, average=None)
        except ValueError:
            print(
                "Average precision requires a sufficient number of samples \
                in a batch which are missing in this sample."
            )
        mean_ap = np.mean(aps)
        return mean_ap
    ```
    </details>

## Instructions to run Action Recognition models
This code was separately tested on RTX 3090, and 3080Ti using CUDA10.2.

1. To prepare the environment, refer to <https://github.com/facebookresearch/SlowFast>. (Please use the earlier version from <https://github.com/facebookresearch/SlowFast/tree/haooooooqi-patch-2> as the latest version in May 2022 does not work). We used Facebook Research's SlowFast code for I3D, SlowFast, X3D.

    * The original code repository can be found:
        * [I3D] <https://github.com/deepmind/kinetics-i3d>
        * [SlowFast] <https://github.com/facebookresearch/SlowFast>
        * [X3D] <https://github.com/facebookresearch/SlowFast>

2. Move and replace files according to the directories in `$DIR_AK/action_recognition/code/code_new`:
    * Helper script to move / create symbolic links to files
        * Remember to change the root directory `$DIR_ROOT` in `$DIR_AK/action_recognition/code/code_new/prepare_dir_AR.sh`
        * `bash $DIR_AK/action_recognition/code/code_new/prepare_dir_AR.sh`

3. Run the model using the following command
    * `python ./tools/run_net.py --cfg configs/AK/SLOWFAST_8x8_R50.yaml`


## Instructions to handle long-tailed distribution 
1. Move and replace files according to the directories in `$DIR_AK/action_recognition/code/code_new`:
    * Helper script to move / create symbolic links to files
        * Remember to change the root directory `$DIR_ROOT` in `$DIR_AK/action_recognition/code/code_new/prepare_dir_AR.sh`
        * `bash $DIR_AK/action_recognition/code/code_new/prepare_dir_AR.sh`
        
    * We used VideoLT's code to handle long-tailed distribution <https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py>

    * The original code repository can be found:
        * [Focal Loss] <https://github.com/facebookresearch/Detectron>
        * [Label-Distribution-Aware Margin Loss (LDAM-DRW)] <https://github.com/kaidic/LDAM-DRW>
        * [Equalization Loss (EQL)] <https://github.com/tztztztztz/eql.detectron2>

* For reference:
    <details><summary>Click to see code</summary>

    ```python script
    import pandas as pd
    import numpy as np

    dir_action_count = '../../data/annot/df_action_count.xlsx'

    class BCELoss(nn.Module):
    '''
    Function: BCELoss
    Params:
        predictions: input->(batch_size, 1004)
        targets: target->(batch_size, 1004)
    Return:
        bceloss
    '''

    def __init__(self,logits=True, reduce="mean"):
        super(BCELoss, self).__init__()
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduce)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduce)

        return BCE_loss

    class FocalLoss(nn.Module):
        '''
        Function: FocalLoss
        Params:
            alpha: scale factor, default = 1
            gamma: exponential factor, default = 0
        Return:
            focalloss
        https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
        Original: https://github.com/facebookresearch/Detectron
        '''

        def __init__(self, logits=True, reduce="mean"):
            super(FocalLoss, self).__init__()
            self.alpha = 1 
            self.gamma = 0 
            self.logits = logits
            self.reduce = reduce

        def forward(self, inputs, targets):
            if self.logits:
                BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            else:
                BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

            if self.reduce == "mean":
                return torch.mean(F_loss)
            elif self.reduce == "sum":
                return torch.sum(F_loss)
            else:
                return F_loss


    class LDAM(nn.Module):
        '''
        https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
        Original: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
        '''

        def __init__(self, logits=True, reduce='mean', max_m=0.5, s=30, step_epoch=80):
            super(LDAM, self).__init__()

            data = pd.read_excel(dir_action_count)
            self.num_class_list = list(map(float, data["count"].tolist()))  
            self.reduce = reduce
            self.logits = logits

            m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.FloatTensor(m_list).cuda()
            self.m_list = m_list
            self.s = s
            self.step_epoch = step_epoch
            self.weight = None

        def reset_epoch(self, epoch):
            idx = epoch // self.step_epoch
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
            self.weight = torch.FloatTensor(per_cls_weights).cuda()

        def forward(self, inputs, targets):
            targets=targets.to(torch.float32)
            batch_m = torch.matmul(self.m_list[None, :], targets.transpose(0, 1))
            batch_m = batch_m.view((-1, 1))
            inputs_m = inputs - batch_m

            output = torch.where(targets.type(torch.uint8), inputs_m, inputs)
            if self.logits:
                loss = F.binary_cross_entropy_with_logits(self.s * output, targets, reduction=self.reduce,
                                                        weight=self.weight)
            else:
                loss = F.binary_cross_entropy(self.s * output, targets, reduction=self.reduce, weight=self.weight)
            return loss


    class EQL(nn.Module):
        '''
        https://github.com/17Skye17/VideoLT/blob/master/ops/losses.py
        Original: https://github.com/tztztztztz/eql.detectron2
        '''

        def __init__(self, logits=True, reduce='mean', max_tail_num=100, gamma=1.76 * 1e-3):
            super(EQL, self).__init__()
            data = pd.read_excel(dir_action_count)
            num_class_list = list(map(float, data["count"].tolist())) 
            self.reduce = reduce
            self.logits = logits

            max_tail_num = max_tail_num
            self.gamma = gamma

            self.tail_flag = [False] * len(num_class_list)
            for i in range(len(self.tail_flag)):
                if num_class_list[i] <= max_tail_num:
                    self.tail_flag[i] = True

        def threshold_func(self):
            weight = self.inputs.new_zeros(self.n_c)
            weight[self.tail_flag] = 1
            weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
            return weight

        def beta_func(self):
            rand = torch.rand((self.n_i, self.n_c)).cuda()
            rand[rand < 1 - self.gamma] = 0
            rand[rand >= 1 - self.gamma] = 1
            return rand

        def forward(self, inputs, targets):
            self.inputs = inputs
            self.n_i, self.n_c = self.inputs.size()

            eql_w = 1 - self.beta_func() * self.threshold_func() * (1 - targets)
            if self.logits:
                loss = F.binary_cross_entropy_with_logits(self.inputs, targets, reduction=self.reduce, weight=eql_w)
            else:
                loss = F.binary_cross_entropy(self.inputs, targets, reduction=self.reduce, weight=eql_w)
            return loss

    _LOSSES = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce": nn.BCELoss,
        "bce_logit": nn.BCEWithLogitsLoss,
        "soft_cross_entropy": SoftTargetCrossEntropy,

        "bce_loss": BCELoss,
        "focal_loss": FocalLoss,
        "LDAM": LDAM,
        "EQL": EQL,
    }

    def get_loss_func(loss_name):
        """
        Retrieve the loss given the loss name.
        Args (int):
            loss_name: the name of the loss to use.
        """
        if loss_name not in _LOSSES.keys():
            raise NotImplementedError("Loss {} is not supported".format(loss_name))
        return _LOSSES[loss_name]
    
    # if __name__ == "__main__":
    #    preds = model(inputs)
    #    loss_fun = get_loss_func(cfg.MODEL.LOSS_FUNC)()
    #    #Uncomment the following line if you want to utilize LDAM-DRW.
    #    # loss_fun.reset_epoch(cur_epoch)
    #    loss = loss_fun(preds, labels)
    ```
</details>

## Solutions to potential issues:
<details><details><summary>Click to expand</summary>

1. TypeError: __init__() got an unexpected keyword argument 'num_sync_devices'
    * Use the earlier version <https://github.com/facebookresearch/SlowFast/tree/haooooooqi-patch-2>

2. Searching for PIL
    Reading https://pypi.org/simple/PIL/
    No local packages or working download links found for PIL
    error: Could not find suitable distribution for Requirement.parse('PIL')

    * https://github.com/facebookresearch/SlowFast/pull/463
    * Change `PIL` to `Pillow` (Line 26) in `$DIR_SLOWFAST/setup.py`
    * Helper script will help replace this file. Remember to run helper script before building slowfast.

3. ImportError: cannot import name 'cat_all_gather' from 'pytorchvideo.layers.distributed' (`$DIR_anaconda_envs`/slowfast/lib/python3.8/site-packages/pytorchvideo/layers/distributed.py)

    * Download and replace the file with <https://github.com/facebookresearch/pytorchvideo/raw/main/pytorchvideo/layers/distributed.py>

4. TypeError: Descriptors cannot not be created directly.
    If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
    If you cannot immediately regenerate your protos, some other possible workarounds are:
    1. Downgrade the protobuf package to 3.20.x or lower.
    2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
    
    More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates    

    * `pip install protobuf==3.20.0`
    * Helper script will help install this. Remember to activate the environment before running the helper script.

5. Configuration description
    * https://github.com/facebookresearch/SlowFast/blob/84cb0ac1780685525aecf51a10cc5ed86ec22705/slowfast/config/defaults.py

</details>

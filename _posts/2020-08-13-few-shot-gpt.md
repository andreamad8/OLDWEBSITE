---
layout: post
title: "Language Model as Few-Shot Learner for Task-Oriented Dialogue Systems"
description: "Evaluating the Few-Shot ability of Language Models in Task-Oriented Dialogue Systems"
comments: false
---
<style>

figcaption {
  /* background-color: black;
  color: white; */
  font-style: italic;
  padding: 2px;
  text-align: center;
}
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  /* width: 70%; */
}
/* CSS Simple Pre Code */
pre {
    background: rgba(197, 225, 184, 0.2);
    /* white-space: pre; */
    /* word-wrap: break-word; */
    overflow: auto;
}

pre.code {
    /* margin: 1px 1px; */
    /* border-radius: 2px; */
    /* border: 1px solid #FDF1DD; */
    position: relative;
}

pre.code label {
    /* font-family: sans-serif; */
    /* font-weight: bold; */
    font-size: 13px;
    /* color: #ddd; */
    position: absolute;
    left: 12px;
    top: 9.5px;
    text-align: center;
    width: 20px;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    pointer-events: none;
}

pre.code code {
    font-family: "Inconsolata","Monaco","Consolas","Andale Mono","Bitstream Vera Sans Mono","Courier New",Courier,monospace;
    display: block;
    margin: 0 0 0 25px;
    /* padding: 1px 16px 14px; */
    /* border-left: 1px solid #555; */
    overflow-x: auto;
    /* font-size: 13px; */
    /* line-height: 19px; */
    /* color: #ddd; */
}




</style>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Modularized Task-Oriented dialogues systems are the core of the current smart speaker generation (e.g. [Alexa](https://en.wikipedia.org/wiki/Amazon_Alexa), [Siri](https://en.wikipedia.org/wiki/Siri) etc.). The main modules of the system are: Natural Language Understanding (NLU), Dialogue State Tracker (DST), Dialogue Policy (DP) and Natural Language Generator (NLG). Each of this module is trained separately using supervised or/and reinforcement learning, and thus a data collection process is usually required. This process for some of the tasks can be laborious and expensive, for example, dialogue policy annotation has to be done by an expert, better if professional linguist. Therefore, having a model that requires only few-sample to actually perform the task is essential. 
<br />
<br />
<img class="center"  width="55%" src="{{ site.url }}/images/IMG.png" alt="...">
<figcaption>Figure 1. Modules in Task-Oriented Dialogue Systems</figcaption>
<br />
<br />
The most successful approach in few-shot learning for task-oriented dialogue systems is [transferring learning](https://en.wikipedia.org/wiki/Transfer_learning), where a large model is firstly pre-trained on a large corpus to be then finetuned on specific tasks. In task-oriented dialogue systems, [Wu et. al 2020](https://arxiv.org/abs/2004.06871), proposed [TOD-BERT](https://github.com/jasonwu0731/ToD-BERT) a large pre-trained model for task-oriented tasks, which can achieve better performance than [BERT](https://arxiv.org/abs/1810.04805) in few-shot NLU, DST and DP. [Liu et. al. 2019](https://arxiv.org/pdf/2004.11727.pdf), instead, proposed a two-step classification for few-shot slot-filling of the NLU module. Similarly, [Peng et. al. 2020](https://arxiv.org/pdf/2002.12328.pdf) proposed a benchmark few-shot NLG and a pre-trained Language Model specialized for the task. Further, [Kale et.al. 2020](https://arxiv.org/pdf/2004.15006v1.pdf) proposed a template rewriting schema based on T5 ([Raffel et al., 2019](https://arxiv.org/abs/1910.10683)) for few-shot NLG in two well-known datasets.

<br />
<br />
<img class="center"  width="35%" src="{{ site.url }}/images/few_shot.png" alt="...">
<figcaption>Figure 2. Language Model priming for few-shot intent recognition. Image inspired by OpenAI GPT-3 ([Brown TB et.al, ‎2020](https://arxiv.org/pdf/2005.14165.pdf))</figcaption>
<br />
<br />
For performing few-shot learning, existing methods requires a set of task-specific parameters since the model is fine-tuned with the few samples. A possible workaround is to perform few-shot learning by priming a [Language Model](https://en.wikipedia.org/wiki/Language_model) (LM) like has been shown in GPT-2 ([Radford, et.al. 2018](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)) and GPT-3 ([Brown TB et.al, ‎2020](https://arxiv.org/pdf/2005.14165.pdf)). In this setting, **NO** parameters are actually updated and thus allowing a single model to perform multiple tasks. In this blog-post, we evaluate the few-shot ability of LM priming on the four task-oriented tasks aforementioned (i.e. NLU, DST, DP, and NLG). 

\\
Currently, GPT-3 is not accessible to the public, or at least not to me now &#128584;, thus we experiment on different GPT-2 sizes model such as SMALL (117M), LARGE (762M). XL (1.45B). All the experiments are run on a single NVIDIA 1080Ti GPU.

### Priming the LM for few-shot learning
Differently from fine-tuning, few-shot learning with Language Models requires to design prefixes to perform few-shot learning ([Radford, et.al. 2018](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf),[Brown TB et.al, ‎2020](https://arxiv.org/pdf/2005.14165.pdf)). In our four tasks, we use three categories of prefix: *binary*, *value-based* and *generative* -- check main paper for more information--. We use different prefixes style depending on the task-oriented tasks and we compare the results of LM few-shot priming with existing fine-tune base models. In all the experiments, we used different amount of shots since different task may fit more or fewer samples in the 1024 max input size.

#### NLU
We use the SNIPS ([Coucke et al., 2018](https://arxiv.org/abs/1805.10190)) dataset for evaluating the *SLOT-FILLING* and *INTENT* recognition task. We follow the few-shot setting of (Liu et.al. 2020) and we use the official CoNLL F1 scorer. For the *INTENT* classification, we finetuned RoBERTa ([Liu et al. 2019](https://arxiv.org/abs/1907.11692)) with 10 samples and we use Accuracy as evaluation metric. We used *value-based* LM prefix for the *SLOT-FILLING* task with a maxim of 15 shots, and *binary* LM prefix for the *INTENT* classification task with a maximum of 10 shots. An example of prefixes for the two tasks is shown in below, and the performance in few-shot is shown in the Figure.

  <pre class='code code-css'>
    <label>SLOT-FILLING</label>
    <code>add tune to my hype playlist => entity_name = none\n
add to playlist confidence boost here comes => entity_name = here comes \n
add the track bg knocc out to the rapcaviar playlist => entity_name =</code>
  </pre>
  <pre class='code code-css'>
    <label>INTENT</label>
    <code>listen to westbam alumb allergic on google music => playmusic = true\n 
rate this novel 4 points out of 6 => playmusic = false\n
add sabrina salerno to the grime instrumentals playlist => playmusic =</code>
  </pre>
<br />
<img class="center"  width="96%" src="{{ site.url }}/images/NLU.png" alt="...">
<!-- <img class="center" width="50%" src="{{ site.url }}/images/INTENT.png" alt="..."> -->

#### DST
We use the MultiWoZ ([Budzianowski et al., 2018](https://arxiv.org/abs/1810.00278)) dataset for evaluating the *DST* task. Differently from other work, we use the last user utterance as input to the model and we update the predicted-DST through turns. For the few-shot evaluation, we follow the setting of ([Wu et. al 2020](https://arxiv.org/abs/2004.06871)) and we report the joint and slot accuracy. As baselines, we use [TOD-BERT](https://github.com/jasonwu0731/ToD-BERT) and [BERT](https://arxiv.org/abs/1810.04805), finetuned with 10% of the training data -- which is equivalent to 500 examples --. We use *value-based* LM prefix, as for the slot-filling task, with a maximum of 15 shots due to limited context.


  <pre class='code code-css'>
    <label>DST</label>
    <code>i need a cab by 12:30 too the contact # and car type will be most helpful => leave_at = 12:30 \n
i would like the taxi to pick me up from the hotel . i need to be at the restaurant at 18:30 . => leave_at = none\n
i would like a taxi from saint john s college to pizza hut fen ditton . => leave_at =</code>
  </pre>
  <img class="center"  width="100%" src="{{ site.url }}/images/DST.png" alt="...">


#### ACT
We use the MultiWoZ ([Budzianowski et al., 2018](https://arxiv.org/abs/1810.00278)) dataset for evaluating the speech *ACT* identification task. Differently from other work, only the system utterance is used as input to the model, instead of including the dialogue history and the user utterance as in ([Wu et. al 2020](https://arxiv.org/abs/2004.06871)). For the few-shot evaluation, we follow the setting of ([Wu et. al 2020](https://arxiv.org/abs/2004.06871)) and we use the F1-score as metric. As baselines, we use [TOD-BERT](https://github.com/jasonwu0731/ToD-BERT) and [BERT](https://arxiv.org/abs/1810.04805), finetuned with 10% of the training data -- which is equivalent to 500 examples --. We use *binary* LM prefix, as for the intent classification task, with a maximum of 15 shots due to limited context.

  <pre class='code code-css'>
    <label>ACT</label>
    <code>yes your booking is successful and your reference number is ri4vvzyc . => offerbooked = true\n
what type of food are you looking for ? => offerbooked = false \n
i do not seem to be finding anything called nusha . what type of food does the restaurant serve ? => offerbooked =</code>
  </pre>
  <img class="center"  width="60%" src="{{ site.url }}/images/ACT.png" alt="...">


#### NLG
We use the FewShotWOZ ([Peng et. al. 2020](https://arxiv.org/pdf/2002.12328.pdf)) dataset for evaluating the *NLG* task. For the few-shot evaluation, we follow the setting of ([Peng et. al. 2020](https://arxiv.org/pdf/2002.12328.pdf)) and we use BLUE and Slot Error Rate as metric. We use SC-LSTM, GPT-2, and SC-GPT-2 ([Peng et. al. 2020](https://arxiv.org/pdf/2002.12328.pdf)) as baselines, all finetuned with 50 examples from the training data. We use *generative* LM prefix with a maximum of 20 shots due to limited context.

  <pre class='code code-css'>
    <label>NLG</label>
    <code>inform(name='hilton san francisco financial district';area='chinatown') => the hilton san francisco financial district is near chinatown\n
inform(name='ocean park motel';dogsallowed='none';phone='4155667020') => the phone number for ocean park motel is 4155667020 . no dogs are allowed there \n
inform(name='super 8 san francisco';phone='8005369326') =></code>
  </pre>
  <img class="center"  width="100%" src="{{ site.url }}/images/NLG.png" alt="...">

### Analysis and Limitation
From the experimental results, we observe that: 
* Larger the model better the performance in both *NLU* and *NLG* tasks, while instead in the *DST* and *ACT* tasks GPT-2 large 762M performs better than the XL (1.54B) version. This is quite uni-intuitive given the results reported in GPT-3. Further investigation would be needed to understand whether changing the prefix can help to improve the performance of larger models;
* In the *NLU*, *ACT* and *NLG*, LM priming few-shot shows promising results, achieving similar or better performance than the weakest finetuning based baseline, which also uses a larger number of shots. On the other hand, in *DST* the gap with the existing baseline is still large.

\\
From the experiment we conducted, we observed two limitations in this approach: 
* using *binary* and *value-based* generation requires as many forward as the number of classes or slots. Although these forward passes are independent, this way achieve few-shots is not as effective as generating directly the class or the tag (e.g. *NLU*). In early experiments, we try to covert all the task into a the *generative* format, thus making the model generate directly the sequence of tags or the class label. Unfortunately, the results in the *generative* setting are bad but we are unsure if larger LMs such as GPT-3 can perform better.
* the current max-input length of GPT-2 (1024 tokens) greatly limit the number of shots that can be provided to the model. Indeed, in most of the tasks not more than 15-shots can be provided, thus making it incomparable with existing models that uses a larger number of shots. 

### Conclusion
In this short blog, we demonstrate the potential of LM priming few-shot learning in the most common task-oriented dialogue systems tasks (NLU, DST, ACT and NLG). Our experiments show that in most of the task larger LMs are better few-shot learner, confirming the hypothesis in ([Brown TB et.al, ‎2020](https://arxiv.org/pdf/2005.14165.pdf)) and, in some cases, they can also achieve similar or better results than the weakest fine-tuning based baseline. Finally, we unveil two limitations of the current LM priming few-shot learning such as computational cost and limited word context size. 

### Acknowledgements
I would like to thanks [Zihan Liu](https://zliucr.github.io/) for helping me with the NLU scorers, [Zhaojiang Lin](https://zlinao.github.io/) for the discussion and insight about the limitation of the LM priming few-shot, [Jason Wu](https://jasonwu0731.github.io/) for providing an easy to use code in ToD-BERT and for clarification about the code and tasks, and [Baolin Peng](https://scholar.google.com/citations?user=u1CNjgwAAAAJ&hl=zh-CN) for the easy to use repository FewShotNLG and for providing help with the scorer. 

### Useful Links
- Github: https://github.com/andreamad8/TASK-ORIENTED-LM-FEWSHOT
- Paper: ARRIVING SOON

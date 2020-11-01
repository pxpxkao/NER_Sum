# Undergraduate Research - Automatic Text Summarization with Deep Learning
數位語音專題研究 - 使用深層學習做文章摘要

## Methods We Tried
- Multi-task learning - a part of model learns NER
- Incorporate entity information with modified attention mechanism
- Entity-Award Embedding - Add named entity information to word embedding


## Multi-Task Learning
### Type
- Soft-Sharing
- Hard Sharing

### Results
| model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------------|:-------------:|:-------------:|:-------------:|
| `transformer` | `6.77` | `1.62` | `1.98` |

### Discussion
- 2 datasets (CoNLL'03 & CNN/DM) differ a lot in vocab (some name entities recognized as OOV)
  - Solution 1: Add more words from NER dataset (CoNLL'03) to dictionary
  - Solution 2: Use tools to tag NER of CNN/DM dataset
- CoNLL’03 dataset too small, may lead to overfitting



## Modified Attention Mechanism

### Proposed Architecture
<img src="images/modified_attn.png" alt="proposed architecture" width="400"/>

### Experiments
- Entity Encoder Type
  - linear
  - transformer (2 layer)
  - MLP (3 layer feed forward network)
- Fusion Type
  - concatenate: 
    - c<sub>t</sub> = [c<sub>t</sub><sup>s</sup> ;c<sub>t</sub><sup>n</sup> ]
  - gated: 
    - g<sub>t</sub> = _&sigma;_ (W<sub>g</sub> c<sub>t</sub><sup>s</sup> + U<sub>g</sub>c<sub>t</sub><sup>n</sup> ) 
    - c<sub>t</sub> = g<sub>t</sub> ⊙ c<sub>t</sub><sup>s</sup> + (1-g<sub>t</sub>) ⊙ c<sub>t</sub><sup>n</sup> 
- NER information
  - added each layer
  - added only at last layer (did not yield good results)

### Results
| model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------------|:-------------:|:-------------:|:-------------:|
| `baseline` | **`37.74`** | `12.67` | `22.28` |
| `linear entity encoder` | `30.63` | `12.55` | `21.17` |
| `MLP entity encoder` | `33.19` | `13.77` | **`23.07`** |
| `transformer entity encoder` | `31.80` | **`13.94`** | `22.25` |
| `gated fusion` | `31.02` | `12.57` | `21.73` |

### Observation & Discussions
- MLP > transformer > linear (entity encoder)
- gated fusion > concatenate 
- sometimes generate repeated sentences
- baseline model tend to copy source
- MLP model knows when to stop 
- Added entity encoder improves ROUGE-2 and -L generally, at the expense of lower ROUGE-1

## Entity-Aware Embedding
### Embedding Indicating Entity Type

### Add Entity Feature at Embedding


### Results

## Future Work
### Coreference Resolution

### Entity Encoder

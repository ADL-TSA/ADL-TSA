**Accuracies of Transformer + Capsule Model vs Capsule Only Model**
|              | Capsule Only | Transformer + Capsule |
|--------------|--------------|-----------------------|
| sentiment140 | 0.7967       | 0.7977                |
| sanders      | 0.7967       | 0.7744                |
| airlines     | 0.7826       | 0.7941                |
| stocks       | 0.5955       | 0.6042                |

**Positive Recall of Transformer + Capsule Model vs Capsule Only Model**
|              | Capsule Only | Transformer + Capsule |
|--------------|--------------|-----------------------|
| sentiment140 | 0.7895       | 0.8099                |
| sanders      | 0.7895       | 0.7750                |
| airlines     | 0.8592       | 0.8732                |
| stocks       | 0.6273       | 0.6122                |

**Negative Recall of Transformer + Capsule Model vs Capsule Only Model**
|              | Capsule Only | Transformer + Capsule |
|--------------|--------------|-----------------------|
| sentiment140 | 0.7334       | 0.7855                |
| sanders      | 0.7334       | 0.7739                |
| airlines     | 0.7629       | 0.7737                |
| stocks       | 0.5400       | 0.5901                |


**F1 Scores of Transformer + Capsule Model vs Capsule Only Model**
|              | Capsule Only | Transformer + Capsule |
|--------------|--------------|-----------------------|
| sentiment140 | 1.2128       | 1.3340                |
| sanders      | 1.2128       | 1.2338                |
| airlines     | 0.8945       | 0.9294                |
| stocks       | 0.9929       | 0.9916                |


**Comparisons of Positive/Negative Recall for Baseline Models (Sanders Dataset)**
|                | Positive Recall | Negative Recall |
|----------------|-----------------|-----------------|
| Trans-CapsNet  | 0.7750          | 0.7739          |
| CapsNet Only   | 0.7895          | 0.7334          |
| BPEF           | 0.693           | 0.786           |
| ChatterBox     | 0.526           | 0.451           |
| FRN            | 0.372           | 0.421           |
| Intridea       | 0.698           | 0.818           |
| LightSIDE      | 0.479           | 0.559           |
| NRC            | 0.425           | 0.355           |
| OpinionFinder  | 0.140           | 0.137           |
| RNTN           | 0.164           | 0.132           |
| ppSentiment140 | 0.551           | 0.368           |
| SentiStrength  | 0.620           | 0.453           |
| Textalytics    | 0.270           | 0.180           |
| Webis          | 0.462           | 0.372           |




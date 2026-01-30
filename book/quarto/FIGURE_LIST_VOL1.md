# Figure List - Volume I: Introduction

_Machine Learning Systems_

**Volume**: 1
**Includes**: Title, Caption, Alt Text

---

## Chapter 1: Introduction

### Figure 1.1: AI Development Timeline

**Caption**: AI Development Timeline: Early AI research focused on symbolic reasoning and rule-based systems, while modern AI leverages data-driven approaches like neural networks to achieve increasingly complex tasks. This progression exposes a shift from hand-coded intelligence to learned intelligence, marked by milestones such as the perceptron, deep blue, and large language models like GPT-3.

**Alt Text**: Timeline from 1950 to 2020 with red line showing AI publication frequency. Gray bands mark two AI Winters (1974-1980, 1987-1993). Callout boxes mark milestones: Turing 1950, Dartmouth 1956, Perceptron 1957, ELIZA 1966, Deep Blue 1997, GPT-3 2020.

### Figure 1.2: AlexNet Convolutional Neural Network Architecture

**Caption**: AlexNet Convolutional Neural Network Architecture: The network that launched the deep learning revolution at ImageNet 2012. With 60 million parameters trained on 1.2 million images across two GTX 580 GPUs, AlexNet achieved 15.3% top-5 error compared to 26.2% for the second-place entry, a 42% relative improvement. The architecture progresses from convolutional layers (green blocks showing spatial feature extraction) through fully connected layers (dense connections), demonstrating that deep networks could automatically learn effective visual features without hand-crafted engineering.

**Alt Text**: 3D diagram of AlexNet with two parallel GPU streams. Green blocks show convolutional layers decreasing from 224x224 input. Red kernels overlay green blocks. Right side shows three dense layers converging to 1000 outputs.

### Figure 1.3: Component Interdependencies

**Caption**: Component Interdependencies: Machine learning system performance relies on the coordinated interaction of models, data, and machines; limitations in any one component constrain the capabilities of the others. Effective system design requires balancing these interdependencies to optimize overall performance and feasibility.

**Alt Text**: Triangle diagram with three circles at vertices labeled Model, Data, and Machine. Double-headed purple arrows connect all three nodes, showing bidirectional dependencies. Icons inside circles depict neural network, database cylinders, and cloud.

### Figure 1.4: Historical Efficiency Trends

**Caption**: Historical Efficiency Trends: The evolution of the DAM Taxonomy components—Data (Information), Algorithm (Logic), and Machine (Physics)—has contributed substantial gains in AI capabilities. Understanding these parallel trajectories helps engineers identify which efficiency dimension currently offers the highest leverage for their workloads.

**Alt Text**: Timeline with three horizontal tracks from 1980 to 2023. Blue track shows Algorithmic Efficiency progressing through Deep Learning Era to Modern Efficiency. Yellow shows Compute Efficiency from General-Purpose through Accelerated to Sustainable Computing. Green shows Data Selection from Scarcity through Big Data to Data-Centric AI.

### Figure 1.5: 

**Alt Text**: Scatter plot showing training efficiency factor from 2012 to 2020. Red dots mark models from AlexNet at 1x to EfficientNet at 44x. Dashed trend line curves upward. Labels identify VGG, ResNet, MobileNet, ShuffleNet versions at their positions.

### Figure 1.6: 

**Alt Text**: Log-scale scatter plot showing petaflop/s-days from 2012 to 2018. Points range from AlexNet at 0.006 to AlphaGoZero at 1900. Dashed diagonal line shows exponential trend. Labels identify models including VGG, ResNet, DeepSpeech2, and AlphaZero.

### Figure 1.7: ML System Lifecycle

**Caption**: ML System Lifecycle: Continuous iteration defines successful machine learning systems, requiring feedback loops to refine models and address performance degradation across data collection, model training, evaluation, and deployment. This cyclical process contrasts with traditional software development and emphasizes the importance of ongoing monitoring and adaptation to maintain system reliability and accuracy in dynamic environments.

**Alt Text**: Flowchart showing cyclical ML lifecycle. Six boxes: Data Collection, Preparation, Model Training, Evaluation, Deployment, Monitoring. Two loops: evaluation returns to preparation; monitoring triggers collection.

### Figure 1.8: 

**Alt Text**: Five pillars diagram: Data Engineering, Training Systems, Deployment Infrastructure, Operations and Monitoring, Ethics and Governance. Pillars rest on foundation labeled Performance Optimization and Hardware Acceleration.

---

## Chapter 2: ML System Architecture

### Figure 2.1: Distributed Intelligence Spectrum

**Caption**: Distributed Intelligence Spectrum: Machine learning system design involves trade-offs between computational resources, latency, and connectivity, resulting in a spectrum of deployment options ranging from centralized cloud infrastructure to resource-constrained edge and TinyML devices. This figure maps these options, highlighting how each approach balances processing location with device capability and network dependence. Source: [@abiresearch2024tinyml].

**Alt Text**: Horizontal spectrum showing 5 deployment tiers from left to right: ultra-low-power devices and sensors, intelligent device, gateway, on-premise servers, and cloud. Arrows indicate TinyML, Edge AI, and Cloud AI spans across the spectrum.

### Figure 2.2: Cloud ML Capabilities

**Caption**: Cloud ML Capabilities: Cloud machine learning systems address challenges related to scale, complexity, and resource management through centralized computing infrastructure and specialized hardware. This figure outlines key considerations for deploying models in the cloud, including the need for reliable infrastructure and efficient resource allocation to handle large datasets and complex computations.

**Alt Text**: Tree diagram with Cloud ML branching to four categories: Characteristics, Benefits, Challenges, and Examples. Each lists items like computational power, scalability, vendor lock-in, and virtual assistants.

### Figure 2.3: 

**Alt Text**: Aerial view of Google Cloud TPU data center with long rows of server racks illuminated by blue LEDs extending toward the horizon across a large facility floor.

### Figure 2.4: Edge ML Dimensions

**Caption**: Edge ML Dimensions: This figure outlines key considerations for edge machine learning, contrasting challenges with benefits and providing representative examples and characteristics. Understanding these dimensions enables designing and deploying effective AI solutions on resource-constrained devices.

**Alt Text**: Tree diagram with Edge ML branching to four categories: Characteristics, Benefits, Challenges, and Examples, listing items like decentralized processing, reduced latency, security concerns, and industrial IoT.

### Figure 2.5: 

**Alt Text**: Collection of IoT devices arranged on a surface: smart home sensors, fitness wearables, environmental monitors, and connected appliances in various sizes and form factors.

### Figure 2.6: Mobile ML Capabilities

**Caption**: Mobile ML Capabilities: Mobile machine learning systems balance performance with resource constraints through on-device processing, specialized hardware acceleration, and optimized frameworks. This figure outlines key considerations for deploying ML models on mobile devices, including the trade-offs between computational efficiency, battery life, and model performance.

**Alt Text**: Tree diagram with Mobile ML branching to four categories: Characteristics, Benefits, Challenges, and Examples. Each lists items like on-device processing, real-time response, battery constraints, and voice recognition.

### Figure 2.7: TinyML System Characteristics

**Caption**: TinyML System Characteristics: Constrained devices necessitate a focus on efficiency, driving trade-offs between model complexity, accuracy, and energy consumption, while enabling localized intelligence and real-time responsiveness in embedded applications. This figure outlines key aspects of TinyML, including the challenges of resource limitations, example applications, and the benefits of on-device machine learning.

**Alt Text**: Tree diagram with TinyML branching to four categories: Characteristics, Benefits, Challenges, and Examples, listing items like low-power operation, always-on capability, resource limitations, and predictive maintenance.

### Figure 2.8: 

**Alt Text**: Small development boards including Arduino Nano BLE Sense and similar microcontroller kits arranged on a surface, each approximately 2-5 cm in length with visible chips and connectors.

### Figure 2.9: ML System Trade-Offs

**Caption**: ML System Trade-Offs: Radar plots quantify performance and operational characteristics across cloud, edge, mobile, and TinyML paradigms, revealing inherent trade-offs between compute power, latency, energy consumption, and scalability. These visualizations enable informed selection of the most suitable deployment approach based on application-specific constraints and priorities.

**Alt Text**: Two radar plots with four overlapping polygons each. Left plot axes: compute power, latency, scalability, energy. Right plot axes: connectivity, privacy, real-time, offline capability.

### Figure 2.10: Deployment Decision Logic

**Caption**: Deployment Decision Logic: This flowchart guides selection of an appropriate machine learning deployment paradigm by systematically evaluating privacy requirements and processing constraints, ultimately balancing performance, cost, and data security. Navigating the decision tree helps practitioners determine whether cloud, edge, mobile, or tiny machine learning best suits a given application.

**Alt Text**: Decision flowchart with four layers: Privacy, Performance, Compute Needs, and Cost. Each layer filters toward deployment options: Cloud ML, Edge ML, Mobile ML, or TinyML based on constraints.

### Figure 2.11: Hybrid System Interactions

**Caption**: Hybrid System Interactions: Data flows upward from sensors through processing layers to cloud analytics for insights, while trained models deploy downward from the cloud to enable inference at the edge, mobile, and TinyML devices. These connection types (deploy, data/results, analyze, and sync) establish a distributed architecture where each paradigm contributes unique capabilities to the overall machine learning system.

**Alt Text**: System diagram with four ML paradigms: TinyML sensors, Edge inference, Mobile processing, and Cloud training. Arrows show deploy, data, results, sync, and assist flows between tiers.

### Figure 2.12: Convergence of ML Systems

**Caption**: Convergence of ML Systems: Diverse machine learning deployments (cloud, edge, mobile, and tiny) share foundational principles in data pipelines, resource management, and system architecture, enabling hybrid solutions and systematic design approaches. Understanding these shared principles allows practitioners to adapt techniques across different paradigms and build cohesive, efficient ML workflows despite varying constraints and optimization goals.

**Alt Text**: Three-layer diagram. Top: Cloud, Edge, Mobile, TinyML implementations. Middle: data pipeline, resource management, architecture principles. Bottom: optimization, operations, trustworthy AI. Arrows connect layers.

---

## Chapter 3: The AI Development Workflow

### Figure 3.1: ML Lifecycle Stages

**Caption**: ML Lifecycle Stages: Two parallel pipelines characterize production ML development. The data pipeline (green, top) progresses from raw collection through ingestion, analysis, labeling, validation, and preparation. The model pipeline (blue, bottom) takes prepared datasets through training, evaluation, validation, and deployment. The prominent feedback arrows emphasize what distinguishes ML from traditional software: monitoring insights continuously inform data refinements, evaluation results trigger model improvements, and deployment experiences reshape collection strategies. These bidirectional flows explain why ML projects average 4-8 iteration cycles before production readiness.

**Alt Text**: Two parallel pipelines: data pipeline (green, top) with 6 stages from collection to preparation; model pipeline (blue, bottom) with 4 stages. Curved feedback arrows connect deployment back to collection and training stages.

### Figure 3.2: Data Scientist Time Allocation

**Caption**: Data Scientist Time Allocation: Data preparation consumes a majority of data science effort, up to 60%, underscoring the need for systematic data engineering practices to prevent downstream model failures and ensure project success. Prioritizing data quality and pipeline development yields greater returns than solely focusing on advanced algorithms. Source: Various industry reports.

**Alt Text**: Pie chart showing data scientist time allocation: 60% cleaning and organizing data, 19% collecting datasets, 9% mining for patterns, 5% building training sets, 4% refining algorithms, 3% other tasks.

### Figure 3.3: Six-Stage ML Lifecycle Framework

**Caption**: Six-Stage ML Lifecycle Framework: A simplified view of ML system development emphasizing sequential progression from problem definition through monitoring. The prominent feedback loop from monitoring back to data collection captures the essential insight: production insights drive continuous refinement across all earlier stages. Unlike traditional software where later phases rarely influence earlier ones, ML systems require this bidirectional flow because data distributions shift, model performance drifts, and operational requirements evolve. Teams that design for this feedback from the start report 2-3x faster iteration cycles than those retrofitting feedback mechanisms.

**Alt Text**: Linear flowchart with 6 boxes: Problem Definition, Data Collection, Model Development, Evaluation, Deployment, Monitoring. Feedback loop arrow curves from Monitoring back to Data Collection.

### Figure 3.4: 

**Alt Text**: Two side-by-side retinal fundus images: left shows healthy retina, right shows diabetic retinopathy with dark red hemorrhage spots scattered across the retina.

### Figure 3.5: ML Lifecycle Dependencies

**Caption**: ML Lifecycle Dependencies: The interconnected feedback paths between stages reveal why ML systems require integrated rather than sequential development. Data gaps identified during evaluation flow back to collection. Validation issues inform training adjustments. Performance insights from monitoring trigger refinements across the pipeline. Deployment constraints propagate backward to influence model design. These dependencies explain why ML technical debt compounds faster than traditional software debt: changes in any stage can cascade through multiple connected stages, requiring coordinated updates rather than isolated fixes.

**Alt Text**: Diagram with 6 boxes: Data Collection, Preparation, Training, Evaluation, Deployment, Monitoring. Labeled feedback arrows show data gaps, validation issues, performance insights, and deployment constraints flowing between stages.

---

## Chapter 4: Data Engineering for ML

### Figure 4.1: Data Quality Cascades

**Caption**: Data Quality Cascades: Errors introduced early in the machine learning workflow amplify across subsequent stages, increasing costs and potentially leading to flawed predictions or harmful outcomes. Recognizing these cascades motivates proactive investment in data engineering and quality control. Source: [@sambasivan2021everyone].

**Alt Text**: Timeline with 7 stages from problem statement to deployment. Colored arcs show errors from data collection propagating to evaluation and deployment stages.

### Figure 4.2: The Four Pillars of Data Engineering

**Caption**: The Four Pillars of Data Engineering: Quality, Reliability, Scalability, and Governance form the foundational framework for ML data systems. Each pillar contributes essential capabilities (solid arrows), while trade-offs between pillars (dashed lines) require careful balancing: validation overhead affects throughput, consistency constraints limit distributed scale, privacy requirements impact performance, and bias mitigation may reduce available training data. Effective data engineering requires managing these tensions systematically rather than optimizing any single pillar in isolation.

**Alt Text**: Four boxes labeled Quality, Reliability, Scalability, and Governance surround a central ML Data System circle. Solid arrows connect each box to center showing contributions; dashed lines between boxes indicate trade-offs.

### Figure 4.3: 

**Alt Text**: Diagram showing voice-activated device with microphone, always-on wake word detector, and connection to main voice assistant that activates upon keyword detection.

### Figure 4.4: Dataset Convergence

**Caption**: Dataset Convergence: Shared datasets can mask limitations and propagate biases across multiple machine learning systems, potentially leading to overoptimistic performance evaluations and reduced generalization to unseen data. Reliance on common datasets creates a false sense of progress within an ecosystem of models, hindering the development of dependable AI applications.

**Alt Text**: Five model boxes labeled A through E at center all connect upward to one central training dataset repository. Arrows downward show shared limitations, biases, and blind spots propagating to all models.

### Figure 4.5: 

**Alt Text**: Historical black-and-white photograph from 1914 showing early traffic control with manual semaphore signals, illustrating how outdated images can appear in modern web scraping results.

### Figure 4.6: Synthetic Data Augmentation

**Caption**: Synthetic Data Augmentation: Combining algorithmically generated data with historical datasets expands training set size and diversity, mitigating limitations caused by scarce or biased real-world data and improving model generalization. This approach enables reliable machine learning system development when acquiring sufficient real-world data is impractical or unethical. Source: AnyLogic [@anylogic_synthetic].

**Alt Text**: Diagram showing historical data icon and simulation cloud icon both feeding into synthetic data generation process, producing an expanded combined training dataset.

### Figure 4.7: Data Pipeline Architecture

**Caption**: Data Pipeline Architecture: Modular pipelines ingest, process, and deliver data for machine learning tasks, enabling independent scaling of components and improved data quality control. Distinct stages (ingestion, storage, and preparation) transform raw data into a format suitable for model training and validation, forming the foundation of reliable ML systems.

**Alt Text**: Three-tier flow diagram with raw data sources and APIs at top, batch and stream ingestion in middle layer, and data warehouse and storage destinations at bottom connected by arrows.

### Figure 4.8: Data Pipeline Architectures

**Caption**: Data Pipeline Architectures: ETL pipelines transform data *before* loading it into a data warehouse, while ELT pipelines load raw data first and transform it within the warehouse, impacting system flexibility and resource allocation for machine learning workflows. Choosing between ETL and ELT depends on data volume, transformation complexity, and the capabilities of the target data storage system.

**Alt Text**: Side-by-side comparison showing ETL pipeline with extract, transform, then load sequence versus ELT pipeline with extract, load, then transform sequence within the data warehouse.

### Figure 4.9: 

**Alt Text**: Two-panel visualization showing raw audio waveform on left transforming into spectrogram on right, with time on horizontal axis and frequency on vertical axis indicated by color intensity.

### Figure 4.10: Data Processing Pipeline

**Caption**: Data Processing Pipeline: A modular end-to-end ML pipeline, as implemented in TensorFlow extended, highlighting key stages from raw data ingestion to trained model deployment and serving. This decomposition enables independent development, versioning, and scaling of each component, improving maintainability and reproducibility of ML systems.

**Alt Text**: Linear flow diagram showing TensorFlow Extended pipeline: data ingestion, validation, transformation, training, evaluation, and deployment stages connected by arrows from left to right.

### Figure 4.11: 

**Alt Text**: Three versions of same street scene showing increasing annotation detail: simple classification label, bounding boxes around vehicles and pedestrians, and pixel-level semantic segmentation with distinct colors.

### Figure 4.12: 

**Alt Text**: Grid of example images showing labeling challenges: blurred animal photos where species is unclear, rare specimens requiring expert knowledge, and ambiguous object boundaries causing annotator disagreement.

### Figure 4.13: AI-Augmented Labeling

**Caption**: AI-Augmented Labeling: Programmatic labeling, distant supervision, and active learning scale data annotation by trading potential labeling errors for increased throughput, necessitating careful system design to balance labeling speed, cost, and model quality. These strategies enable machine learning systems to overcome limitations imposed by manual annotation alone, facilitating deployment in data-scarce environments. Source: Stanford AI Lab.

**Alt Text**: Hierarchical diagram with question about getting labeled data at top. Four branches: traditional supervision, semi-supervised, weak supervision, and transfer learning. Active learning branches as cost-saving alternative.

### Figure 4.14: 

**Alt Text**: Pipeline showing audio waveform and text transcript inputs processed through forced alignment stage, then segmented into individual one-second labeled keyword samples for KWS training.

### Figure 4.15: Data Governance Pillars

**Caption**: Data Governance Pillars: Robust data governance establishes ethical and reliable machine learning systems by prioritizing privacy, fairness, transparency, and accountability throughout the data lifecycle. These interconnected pillars address unique challenges in ML workflows, ensuring responsible data usage and auditable decision-making processes.

**Alt Text**: Central stacked database icon surrounded by four governance elements: privacy shield, security lock, compliance checklist, and transparency document. Gear icons show interconnections between all elements.

### Figure 4.16: Data Governance Documentation

**Caption**: Data Governance Documentation: Data cards standardize critical dataset information, enabling transparency and accountability required for regulatory compliance with laws like GDPR and HIPAA. By providing a structured overview of dataset characteristics, intended uses, and potential risks, data cards facilitate responsible AI practices and support data subject rights.

**Alt Text**: Sample data card template showing structured fields: dataset name and description at top, authorship and funding details in middle sections, and intended uses with potential risks at bottom.

### Figure 4.17: Data Pipeline Debugging Flowchart

**Caption**: Data Pipeline Debugging Flowchart: A systematic diagnostic process for identifying the root causes of ML system failures. By asking sequential questions about degradation patterns, training-validation gaps, and production discrepancies, engineers can isolate specific data issues—from drift and skew to label noise and bias—before investigating model architecture.

**Alt Text**: Vertical flowchart with four blue diamond decision nodes and red result boxes. Top diamond asks if accuracy degrades over time, leading to Data Drift result. Second asks if training accuracy exceeds validation, leading to Overfitting. Third asks if validation exceeds production accuracy, leading to Training-Serving Skew. Fourth asks about subgroup inconsistency, leading to Bias. Gray box at bottom shows Model Architecture issue if all answers are no.

---

## Chapter 5: Deep Learning Systems Foundations

### Figure 5.1: 

**Alt Text**: Nested circles diagram showing AI as outermost circle containing Machine Learning, which contains Deep Learning, which contains Neural Networks at the center. Arrows indicate progression from broad AI concepts to specific neural network implementations.

### Figure 5.2: Rule-Based System

**Caption**: Rule-Based System: Traditional programming relies on explicitly defined rules to map inputs to outputs, limiting adaptability to complex or uncertain environments as every possible scenario must be anticipated and coded. This approach contrasts with deep learning, where systems learn patterns from data instead of relying on pre-programmed logic.

**Alt Text**: Breakout game grid with 3 rows of 5 colored bricks at top, brown paddle at bottom, and ball with trajectory arrow. Code snippet shows explicit if-then rules for collision detection: removeBrick, update ball velocity.

### Figure 5.3: Rule-Based Programming

**Caption**: Rule-Based Programming: Traditional programs operate on data using explicitly defined rules, forming the basis for early AI systems but lacking the adaptability of modern machine learning approaches. This approach contrasts with deep learning, where the system infers rules from examples rather than relying on pre-programmed logic.

**Alt Text**: Flow diagram with three boxes: Rules and Data as inputs flowing into central Traditional Programming box, which outputs Answers. Arrows show data flow direction from inputs to output.

### Figure 5.4: 

**Alt Text**: Decision tree flowchart for activity classification. Branches split on conditions like speed less than 4 mph leading to walking, 4-15 mph to running, greater than 15 mph to biking. Additional branches handle edge cases and transitions.

### Figure 5.5: 

**Alt Text**: Three-panel image showing HOG feature extraction: original grayscale photo of person on left, gradient magnitude visualization in center, and HOG descriptor grid overlay on right showing edge orientation histograms per cell.

### Figure 5.6: Data-Driven Rule Discovery

**Caption**: Data-Driven Rule Discovery: Deep learning models learn patterns and relationships directly from data, eliminating the need for manually specified rules and enabling automated feature extraction from raw inputs. This contrasts with traditional programming, where both rules and data are required to generate outputs, and classical machine learning, where rules are inferred from labeled data.

**Alt Text**: Flow diagram with three boxes: Answers and Data as inputs flowing into central Machine Learning box, which outputs Rules. Arrows show inverted flow compared to traditional programming, with rules as output rather than input.

### Figure 5.7: 

**Alt Text**: Side-by-side comparison of biological neuron and artificial neuron. Left shows biological cell with dendrites, cell body, and axon. Right shows mathematical model with inputs x, weights w, summation node, activation function, and output. Arrows map corresponding components between the two.

### Figure 5.8: 

**Alt Text**: Log-scale scatter plot showing training compute in FLOPS from 1950 to 2022. Points represent AI models, with different colors for pre-deep-learning era, deep learning era, and large-scale models. Trend lines show 1.4x growth before 2010 and 3.4-month doubling after 2012.

### Figure 5.9: The Virtuous Cycle of Deep Learning Progress

**Caption**: The Virtuous Cycle of Deep Learning Progress: Three mutually reinforcing factors drove the deep learning revolution. Increased data availability enabled training larger models, algorithmic innovations improved learning efficiency and model architectures, and advances in computing infrastructure provided the processing power to handle growing computational demands. Each breakthrough in one area created opportunities in the others, accelerating progress across the field.

**Alt Text**: Three connected boxes in a cycle: green Data Availability flows to blue Algorithmic Innovations, which flows to red Computing Infrastructure, which loops back to Data Availability. Yellow background box labeled Key Breakthroughs contains all three elements.

### Figure 5.10: Perceptron Architecture

**Caption**: Perceptron Architecture: The fundamental computational unit of neural networks, showing inputs multiplied by weights, summed with bias, and passed through an activation function to produce output.

**Alt Text**: Perceptron diagram with inputs x1 through xi on left, each connected to weight circles w1j through wij. Weights feed into red summation node, which receives bias b from below. Output z flows to blue sigma activation function box, producing output y-hat on right.

### Figure 5.11: Common Activation Functions

**Caption**: Common Activation Functions: Neural networks rely on nonlinear activation functions to approximate complex relationships. Each function exhibits distinct characteristics: sigmoid maps inputs to $(0,1)$ with smooth gradients, tanh provides zero-centered outputs in $(-1,1)$, ReLU introduces sparsity by outputting zero for negative inputs, and softmax converts logits into probability distributions. These different behaviors enable networks to learn different types of patterns and relationships.

**Alt Text**: Four plots arranged in 2x2 grid. Top-left: Sigmoid S-curve from 0 to 1. Top-right: Tanh S-curve from -1 to 1. Bottom-left: ReLU showing zero for negative x, linear for positive x. Bottom-right: Softmax showing exponential curve approaching small positive values.

### Figure 5.12: Non-Linear Activation

**Caption**: Non-Linear Activation: Neural networks model complex relationships by applying non-linear activation functions to weighted sums of inputs, enabling the representation of non-linear decision boundaries. These functions transform input values, creating the capacity to learn patterns beyond linear combinations via the arrangement of points.

**Alt Text**: Two scatter plots side by side. Left plot shows cyan and green points with straight red line failing to separate them, labeled NN without Activation Function. Right plot shows same points with curved red decision boundary successfully separating classes, labeled NN with Activation Function.

### Figure 5.13: Layered Network Architecture

**Caption**: Layered Network Architecture: Deep neural networks transform data through successive layers, enabling the extraction of increasingly complex features and patterns. Each layer applies non-linear transformations to the outputs of the previous layer, ultimately mapping raw inputs to desired outputs.

**Alt Text**: Neural network diagram showing input layer on left with multiple nodes, two hidden layers in middle with interconnected nodes, and output layer on right. Arrows show data flow from left to right through fully connected layers.

### Figure 5.14: Forward Propagation Process

**Caption**: Forward Propagation Process: Neural networks transform input data into predictions by sequentially applying weighted sums and activation functions across interconnected layers, enabling complex pattern recognition. This layered computation forms the basis for both making inferences and updating model parameters during training.

**Alt Text**: Two panels showing MNIST digit recognition. Panel a: 28x28 pixel image of digit 7 connected to hidden layer circles, then to 10 output nodes with one highlighted for digit classification. Panel b: Same architecture with flattened 784-pixel vector representation of input image.

### Figure 5.15: Training Loop Architecture

**Caption**: Training Loop Architecture: Complete neural network training flow showing forward propagation through layers to generate prediction, comparison with true value via loss function, and backward propagation of gradients through optimizer to update weights and biases.

**Alt Text**: Neural network training diagram. Left side shows input X flowing through blue, red, and green node layers via forward propagation (red arrow). Right side shows prediction and true value boxes feeding into loss function, which outputs loss score to optimizer, which updates weights and biases. Orange arrow shows backward propagation path.

### Figure 5.16: Inference vs. Training Flow

**Caption**: Inference vs. Training Flow: During inference, neural networks utilize learned weights for forward pass computation only, simplifying the data flow and reducing computational cost compared to training, which requires both forward and backward passes for weight updates. This streamlined process enables efficient deployment of trained models for real-time predictions.

**Alt Text**: Two parallel diagrams comparing inference and training. Both show stacked rectangles representing batches feeding into network layers and output nodes. Inference section shows smaller varied batch sizes with dashed outlines. Training section shows larger fixed batches with solid outlines. Network architecture identical in both with fully connected layers.

### Figure 5.17: 

**Alt Text**: Grid of handwritten digit samples from USPS dataset showing digits 0-9 in multiple rows. Each digit appears in several variations demonstrating different handwriting styles, stroke widths, slants, and character formations that OCR systems must recognize.

### Figure 5.18: USPS Inference Pipeline

**Caption**: USPS Inference Pipeline: The mail sorting system combines traditional computing (green) with neural network inference (blue) in a sequential pipeline. Raw envelope images undergo preprocessing (thresholding, segmentation, normalization) before the neural network classifies individual digits. Post-processing then applies confidence thresholds and formats sorting instructions. This hybrid architecture—where the neural network handles pattern recognition while conventional code handles data preparation and decision logic—remains the standard template for production ML systems.

**Alt Text**: Linear pipeline with 6 boxes connected by arrows. From left: Raw Input and Pre-processing in green Traditional Computing section, Neural Network in orange Deep Learning section, then Raw Output, Post-processing, and Final Output in green Traditional Computing section.

---

## Chapter 6: DNN Architectures

### Figure 6.1: Multi-Layer Perceptron Architecture

**Caption**: Multi-Layer Perceptron Architecture: Dense connectivity where every neuron connects to all neurons in adjacent layers, implemented through matrix multiplications. For MNIST classification, a 784-dimensional input connects to 100 hidden neurons through a $784 \times 100$ weight matrix, requiring 78,400 multiply-accumulate operations per sample. The highlighted neuron receives weighted contributions from all input pixels, demonstrating the $O(N \times M)$ computational complexity inherent to fully-connected layers. Adapted from [@reagen2017deep].

**Alt Text**: Three-layer neural network with 4 input nodes, 5 hidden nodes, and 2 output nodes. Lines connect every node to all nodes in adjacent layers. One highlighted node shows weighted connections from all inputs, demonstrating dense O(N x M) connectivity.

### Figure 6.2: Spatial Feature Extraction

**Caption**: Spatial Feature Extraction: Convolutional neural networks identify patterns independent of their location in an image by applying learnable filters across the input, enabling robust object recognition. These filters detect local features, and their repeated application across the image creates translation invariance, the ability to recognize a pattern regardless of its position.

**Alt Text**: Two identical zebra images at different positions in input frames. Arrows show same filter applied to both, producing matching feature activations. Demonstrates translation invariance: detecting patterns regardless of spatial position in image.

### Figure 6.3: Attention Weights Visualization

**Caption**: Attention Weights Visualization: Attention head (layer 4, head 2) resolving the pronoun "they" in the sentence. Line thickness indicates attention weight magnitude: "student" receives the highest attention (bold connection), followed by "The" and "finish", demonstrating that attention learns to link pronouns with their referents across arbitrary distances. This dynamic routing replaces RNN sequential processing with $O(1)$ information flow depth, enabling parallel computation across all 12 positions simultaneously.

**Alt Text**: Sentence tokens listed vertically with cyan attention lines from highlighted word they connecting to all other tokens. Thick lines to student and finish show high attention weights. Demonstrates pronoun-referent linking across arbitrary distances.

### Figure 6.4: Query-Key-Value Attention Mechanism

**Caption**: Query-Key-Value Attention Mechanism: For a 6-token sequence, queries (cyan) match against keys (red) to produce a $6 \times 6$ attention matrix with $O(N^2)$ entries. Color intensity indicates attention weight: darker cells show stronger relationships. Each output position aggregates information from all values (green) weighted by its attention row. The matrix structure reveals both the computational pattern (36 similarity computations) and the memory bottleneck (storing $N^2$ attention weights). Source: Transformer Explainer [@transformer_explainer].

**Alt Text**: 6x6 attention matrix with gradient coloring from blue to red indicating attention weights. Cyan query vectors enter from left, red key vectors from top, green value vectors from below. Output vectors exit right, showing weighted aggregation pattern.

### Figure 6.5: QKV Projection Computation

**Caption**: QKV Projection Computation: The embedding matrix $(6 \times 768)$ multiplies with QKV weight matrices $(768 \times 2304)$ plus bias to produce combined projections $(6 \times 2304)$. The 2304 output dimension contains concatenated query, key, and value projections (each 768-dimensional). This single batched matrix multiplication, requiring $6 \times 768 \times 2304 = 10.6$ million MACs, replaces three separate projection operations for efficiency. Modern implementations fuse this with subsequent attention computation to reduce memory bandwidth requirements. Source: Transformer Explainer [@transformer_explainer].

**Alt Text**: Matrix multiplication: 6x768 embedding times 768x2304 QKV weights plus 2304 bias equals 6x2304 output. Blue and red regions show concatenated query, key, value projections. Token labels Data, visualization, em, powers, users, to.

### Figure 6.6: Transformer Architecture (Encoder-Decoder)

**Caption**: Transformer Architecture (Encoder-Decoder): Complete architecture from Vaswani et al. The encoder (left, repeated $N$ times) consists of multi-head attention followed by feed-forward layers, each with residual connections (arrows bypassing blocks) and layer normalization. The decoder (right) adds masked attention to prevent attending to future tokens during autoregressive generation. Positional encodings (sine waves) inject sequence order information absent from the permutation-invariant attention operation. This design enables training parallelism across all positions while the decoder maintains autoregressive causality during inference. Source: Vaswani et al. [@vaswani2017attention].

**Alt Text**: Encoder-decoder architecture. Encoder: multi-head attention, add-norm, feed-forward, add-norm, repeated Nx. Decoder adds masked attention. Positional encoding sine waves at inputs. Skip connections bypass sublayers. Linear and softmax at top.

### Figure 6.7: im2col Transformation

**Caption**: im2col Transformation: Converts convolution to GEMM by rearranging image patches into columns. The input feature maps (cyan/orange grids, $3 \times 3$) are unfolded so each sliding window position becomes a matrix column, while filter kernels (green/yellow, $2 \times 2$) become rows. The resulting $4 \times 8$ matrix multiplication produces all output positions in one operation. This transformation trades 2x memory overhead (duplicating overlapping pixels) for 5-10x speedup by leveraging decades of BLAS optimizations and enabling efficient GPU parallelization.

**Alt Text**: Left: two 3x3 input feature maps in cyan and orange. Center: 4x8 transformed matrix with unfolded patches as columns. Right: 8x1 filter kernel vector. Red boxes highlight how sliding windows become matrix columns for GEMM.

### Figure 6.8: Data Movement Primitives

**Caption**: Data Movement Primitives: Four fundamental patterns govern information flow in neural network computation. Broadcast (top-left) replicates a single value to all destinations, used when sharing weights across batch elements. Scatter (top-right) distributes distinct elements to different destinations, enabling work partitioning. Gather (bottom-left) collects distributed values to a single location, as in attention pooling. Reduction (bottom-right) combines multiple values through aggregation (sum, max), appearing in gradient synchronization and attention scoring. Moving data typically costs 100-1000x more energy than computation, making these patterns critical optimization targets.

**Alt Text**: Four diagrams with nodes and arrows. Broadcast: one red square to four nodes. Scatter: four colored squares to four nodes. Gather: four nodes with colored squares to one. Reduction: four colored nodes combine through aggregation to one.

### Figure 6.9: Architecture Selection Decision Framework

**Caption**: Architecture Selection Decision Framework: A systematic flowchart for choosing neural network architectures based on data characteristics and deployment constraints. The process begins with data type identification (text/sequences/images/tabular) to select initial architecture candidates (Transformers/RNNs/CNNs/MLPs), then iteratively evaluates memory budget, computational cost, inference speed, accuracy targets, and hardware compatibility.

**Alt Text**: Flowchart from Define Problem branching by data type to Transformers, RNNs, CNNs, or MLPs. Diamond nodes check memory, compute, speed, accuracy, deployment. No paths loop to scale down or increase capacity. Yes path leads to selected.

---

## Chapter 7: AI Frameworks

### Figure 7.1: Computational Library Evolution

**Caption**: Computational Library Evolution: Modern machine learning frameworks build upon decades of numerical computing advancements, transitioning from low-level routines like BLAS and LAPACK to high-level abstractions in NumPy, SciPy,[^fn-scipy-date] and finally to deep learning frameworks such as TensorFlow and PyTorch. This progression reflects a shift toward increased developer productivity and accessibility in machine learning system development.

**Alt Text**: Horizontal timeline from 1979 to 2018 with colored boxes marking key years. Dashed arrows connect to milestones below: 1979 BLAS introduced, 1992 LAPACK extends BLAS, 2006 NumPy becomes Python's numerical backbone, 2007 SciPy and Theano introduce computational graphs, 2015 TensorFlow revolutionizes distributed ML, 2016 PyTorch introduces dynamic graphs, 2018 JAX introduces functional paradigms.

### Figure 7.2: Computational Graph

**Caption**: Computational Graph: Directed acyclic graphs represent machine learning models as a series of interconnected operations, enabling efficient computation and automatic differentiation. This example presents a simple computation, $z = x \\times y$, where nodes define operations and edges specify the flow of data between them.

**Alt Text**: Simple directed graph with nodes x and y flowing into function f(x,y) which outputs z.

### Figure 7.3: Neural Network Computation Graph

**Caption**: Neural Network Computation Graph: This diagram represents a neural network as a directed acyclic graph, where nodes denote operations and edges represent data flow. The system components (memory management, device placement) interact with the computational graph to optimize resource allocation before execution.

**Alt Text**: Left side shows computational graph with 6 operation nodes connected by data flow edges. Right side shows system components box with Memory Management and Device Placement nodes that interact with the computational graph.

### Figure 7.4: Dynamic Graph Execution Flow

**Caption**: Dynamic Graph Execution Flow: In eager execution, each operation is defined and immediately executed before the next operation begins. This define-by-run model enables natural debugging and data-dependent control flow at the cost of optimization opportunities.

**Alt Text**: Flow diagram showing Start to Operation 1 to Operation 1 Executed to Operation 2 to Operation 2 Executed to End. Above arrows show Define Operation, Execute Operation, Define Next Operation, Execute Operation, Repeat Until Done.

### Figure 7.5: Static Computation Graph

**Caption**: Static Computation Graph: Machine learning frameworks first define computations as a graph of operations, enabling global optimizations like operation fusion and efficient resource allocation before any data flows through the system. This two-phase approach separates graph construction and optimization from execution, improving performance and predictability.

**Alt Text**: Flow diagram showing two phases. Definition Phase: Define Operations, Declare Variables, Build Graph. Execution Phase: Load Data, Run Graph, Get Results. Arrows connect boxes left to right.

### Figure 7.6: The Compilation Continuum

**Caption**: The Compilation Continuum: Optimal execution strategy depends on development-to-production ratio. Left region (high dev iterations): eager mode dominates. Right region (high prod executions): compilation dominates. The crossover point depends on compilation cost and per-execution speedup.

**Alt Text**: Graph with x-axis 'Production Executions' (log scale) and y-axis 'Total Time'. Three lines: Eager (steep slope), JIT (moderate slope with offset), Static (gentle slope with larger offset). Lines cross at different points showing when compilation becomes beneficial.

### Figure 7.7: Three-Dimensional Tensor

**Caption**: Three-Dimensional Tensor: Higher-rank tensors extend the concepts of scalars, vectors, and matrices by arranging data in nested structures; this figure represents a three-dimensional tensor as a stack of matrices, enabling representation of complex, multi-dimensional data relationships. Tensors with rank greater than two are fundamental to representing data in areas like image processing and natural language processing, where data possesses inherent multi-dimensional structure.

**Alt Text**: Four shapes showing tensor ranks left to right: single box labeled Rank 0, vertical column of numbers labeled Rank 1, 2D grid of numbers labeled Rank 2, and 3D cube labeled Rank 3.

### Figure 7.8: Multidimensional Data Representation

**Caption**: Multidimensional Data Representation: Images naturally map to tensors with dimensions representing image height, width, and color channels, forming a three-dimensional array; stacking multiple images creates a fourth dimension for batch processing and efficient computation. *credit: niklas lang [https://towardsdatascience.com/what-are-tensors-in-machine-learning-5671814646ff](https://towardsdatascience.com/what-are-tensors-in-machine-learning-5671814646ff)*.

**Alt Text**: Three stacked 3x3 grids in red, green, and blue representing RGB color channels. Dimension labels show width 3 pixels, height 3 pixels, and 3 color channels forming a 3D tensor for image data.

### Figure 7.9: Tensor Memory Layout

**Caption**: Tensor Memory Layout: A 2×3 tensor can be stored in linear memory using either row-major (C-style) or column-major (Fortran-style) ordering. Strides define the number of elements to skip in each dimension when moving through memory, enabling frameworks to calculate memory addresses for tensor[i,j] as base_address + i×stride[0] + j×stride[1]. The choice of memory layout significantly impacts cache performance and computational efficiency.

**Alt Text**: Left: 2x3 tensor grid with values 1-6. Right: two linear arrays showing row-major layout (1,2,3,4,5,6) and column-major layout (1,4,2,5,3,6). Below: stride calculations for row-major [3,1] and column-major [1,2].

### Figure 7.10: 3D Parallelism

**Caption**: 3D Parallelism: Training can be parallelized across multiple dimensions including data batches, pipeline stages, and model partitions. This figure illustrates how large-scale training systems distribute computation across a grid of accelerators. These distributed training strategies are covered in advanced distributed systems literature.

**Alt Text**: Grid of 8 GPU clusters in 2 rows and 4 columns. Each cluster contains 4 stacked cubes. Colors vary: blue, red, green, orange in bottom row; olive, yellow, brown, pink in top row.

### Figure 7.11: Framework Operational Hierarchy

**Caption**: Framework Operational Hierarchy: Machine learning frameworks abstract hardware complexities through layered operations (scheduling, memory management, and resource optimization), enabling efficient execution of mathematical models on diverse computing platforms. This hierarchical structure transforms high-level model descriptions into practical implementations by coordinating resources and managing computations.

**Alt Text**: Three grouped boxes connected by arrows. System-Level: Scheduling, Memory Management, Resource Optimization. Numerical: GEMM, BLAS, Element-wise Operations. Hardware: Kernel Management, Memory Abstraction, Execution Control.

### Figure 7.12: TensorFlow 2.0 Architecture

**Caption**: TensorFlow 2.0 Architecture: This diagram outlines TensorFlow's modular design, separating eager execution from graph construction for increased flexibility and ease of debugging. TensorFlow core provides foundational APIs, while Keras serves as its high-level interface for simplified model building and training, supporting deployment across various platforms and hardware accelerators. Source: [TensorFlow.](https://blog.tensorflow.org/2019/01/whats-coming-in-tensorflow-2-0.html).

**Alt Text**: Two-column diagram. Training: data preprocessing, tf.keras, TensorFlow Hub, Premade Estimators, Distribution Strategy across CPU/GPU/TPU. Deployment via SavedModel to TensorFlow Serving, Lite, JS, and language bindings.

### Figure 7.13: 

**Alt Text**: Hub diagram with ONNX logo at center. Left side: PyTorch, TensorFlow, Keras with arrows pointing inward. Right side: TF Lite, ONNX Runtime with arrows outward.

---

## Chapter 8: AI Training

### Figure 8.1: Activation Function Performance

**Caption**: Activation Function Performance: CPU execution time varies significantly across common activation functions, with tanh and relu offering substantial speed advantages over sigmoid on this architecture. These differences impact system-level considerations such as training time and real-time inference capabilities, guiding activation function selection for performance-critical applications.

**Alt Text**: Bar chart comparing CPU execution times: Sigmoid at 1.1 seconds, Tanh at 0.61 seconds, ReLU at 0.78 seconds, and Softmax at 0.91 seconds.

### Figure 8.2: Training Roofline Model

**Caption**: Training Roofline Model: Performance of training operations plotted against arithmetic intensity. The sloped region (left of ridge point) represents memory-bound operations where performance scales with bandwidth; the flat region (right) represents compute-bound operations achieving peak throughput. GPT-2 matrix multiplications operate in the compute-bound regime, while normalization and activation operations are memory-bound. FlashAttention shifts standard attention from below to above the ridge point.

**Alt Text**: Log-log plot showing roofline model with memory-bound slope and compute-bound ceiling. Points show different training operations: MatMul above ridge point, LayerNorm and Softmax below. Arrow shows FlashAttention improvement.

### Figure 8.3: Pipeline Architecture

**Caption**: Pipeline Architecture: Machine learning systems organize training through interconnected data, training, and evaluation pipelines, enabling iterative model refinement and performance assessment. Data flows sequentially through these components, with evaluation metrics providing feedback to guide the training process and ensure reproducible results.

**Alt Text**: Block diagram with three connected boxes: Data Pipeline, Training Loop, and Evaluation Pipeline. Arrows show data flow with feedback from evaluation.

### Figure 8.4: GPU-Accelerated Training

**Caption**: GPU-Accelerated Training: Modern deep learning relies on gpus to parallelize matrix operations, significantly accelerating the forward and backward passes required for parameter updates during training. This single-GPU workflow iteratively refines model parameters by computing gradients from loss functions and applying them to minimize prediction errors.

**Alt Text**: Neural network diagram showing data cylinders feeding into a network of connected nodes. A GPU box at bottom processes the forward and backward pass computations.

### Figure 8.5: Data Pipeline Architecture

**Caption**: Data Pipeline Architecture: Modern machine learning systems utilize pipelines to efficiently move data from storage to gpus for parallel processing, enabling faster model training and inference. This diagram presents a typical pipeline with stages for formatting, preprocessing, batching, and distributing data across multiple GPU workers.

**Alt Text**: Block diagram showing data flow through three zones: Storage Zone with raw data, CPU Preprocessing Zone with format, process, and batch stages, and GPU Training Zone with three GPU workers.

### Figure 8.6: Memory Footprint Breakdown

**Caption**: Memory Footprint Breakdown: Large language models require substantial memory, with optimizer states and gradients often exceeding the size of model weights themselves. This figure quantifies the memory usage of the llama-7B model, revealing how techniques like compression can significantly reduce the overall footprint by minimizing the storage requirements for optimizer data.

**Alt Text**: Stacked horizontal bar chart comparing memory usage across four optimizers for LLaMA-7B. Shows components: others, weight gradient, optimization, activation, and weight. Dashed red line marks RTX 4090 memory limit at 30 GB.

### Figure 8.7: 

**Alt Text**: TensorFlow profiler screenshot showing GPU activity timeline. Colored blocks indicate computation periods with white gaps revealing idle time when GPU waits for data loading to complete.

### Figure 8.8: Training Optimization Decision Flowchart

**Caption**: Training Optimization Decision Flowchart: Systematic approach to optimization selection based on profiling results. Begin by measuring GPU utilization, then follow the decision path to identify whether the bottleneck is data-bound, memory-bound, or compute-bound. Each path leads to specific techniques that address the identified constraint.

**Alt Text**: Flowchart showing optimization decision tree starting from Profile Training Run, branching based on GPU utilization and memory pressure to different optimization techniques.

### Figure 8.9: Sequential Data Transfer

**Caption**: Sequential Data Transfer: Standard data fetching pipelines execute transfers from disk to CPU, CPU to GPU, and through GPU processing one at a time, creating bottlenecks and limiting computational throughput during model training. This serial approach prevents overlapping computation and data movement, hindering efficient resource utilization.

**Alt Text**: Gantt chart showing sequential data pipeline over two epochs. Four rows: Open, Read, Train, and Epoch. Operations execute serially with gaps between phases, spanning from 00:00 to 01:30.

### Figure 8.10: Pipeline Parallelism

**Caption**: Pipeline Parallelism: Overlapping computation and data fetching reduces overall job completion time by concurrently processing data and preparing subsequent batches. This optimization achieves a 40% speedup, finishing in 40 s compared to 90 s with naive sequential fetching.

**Alt Text**: Gantt chart showing optimized pipeline with overlapping operations. Read and Train execute in parallel across time slices. Two epochs complete in approximately 55 seconds total.

### Figure 8.11: Mixed Precision Training

**Caption**: Mixed Precision Training: Reduced precision formats (FP16, bfloat16) accelerate deep learning by decreasing memory bandwidth and computational requirements during both forward and backward passes. Master weights stored in FP32 precision accumulate updates from reduced precision gradients, preserving accuracy while leveraging performance gains from lower precision arithmetic.

**Alt Text**: Flowchart showing 7-step mixed precision training cycle. FP32 master weights convert to FP16 for forward pass, loss scaling protects gradients during backpropagation, then gradients update FP32 weights.

### Figure 8.12: Gradient Accumulation

**Caption**: Gradient Accumulation: Effective batch size increases without increasing per-step memory requirements by accumulating gradients from multiple micro-batches before updating model parameters, simulating training with a larger batch. Note that gradient accumulation can affect Batch Normalization behavior since statistics are computed on micro-batches rather than the full effective batch. This technique enables training with large models or datasets when memory is limited, improving training stability and potentially generalization performance.

**Alt Text**: Block diagram showing three batches computing individual losses and gradients. Arrows flow from Batch 1, 2, 3 through Losses to Gradients boxes, then combine into a single summed gradient output.

### Figure 8.13: Activation Checkpointing

**Caption**: Activation Checkpointing: Trading memory usage for recomputation during backpropagation enables training deeper neural networks. By storing only a subset of activations from the forward pass and recomputing others on demand, this technique reduces peak memory requirements at the cost of increased training time.

**Alt Text**: Two-row diagram showing activation checkpointing. Top row: forward pass with checkpointed nodes (filled) and discarded nodes (dashed). Bottom row: backward pass recomputing discarded activations from checkpoints.

### Figure 8.14: Computing System Evolution

**Caption**: Computing System Evolution: Hardware advancements continuously adapted to the increasing demands of machine learning workloads, transitioning from centralized mainframes to specialized architectures optimized for parallel processing and massive datasets.

**Alt Text**: Timeline spanning 1950s to 2020s showing evolution from mainframes through HPC and warehouse-scale computing to AI hypercomputing with GPUs and TPUs.

### Figure 8.15: Data Parallelism

**Caption**: Data Parallelism: Each GPU holds a complete model copy, processes different data batches, then synchronizes gradients. This approach scales training throughput linearly with GPU count when models fit in single-GPU memory.

**Alt Text**: Diagram showing input data splitting into 4 batches, each assigned to a GPU for forward/backward pass, with gradients aggregating for model update.

### Figure 8.16: Model Parallelism

**Caption**: Model Parallelism: The model is partitioned across devices, with intermediate activations passing between them. This enables training models larger than single-GPU memory at the cost of sequential dependencies.

**Alt Text**: Diagram showing input flowing through model parts on different devices, with forward pass going right and backward pass returning left.

### Figure 8.17: Layer-wise Partitioning

**Caption**: Layer-wise Partitioning: A 24-layer transformer distributed across four devices, with each device responsible for six consecutive transformer blocks. Communication occurs only at partition boundaries.

**Alt Text**: Diagram showing transformer blocks 1-6 on GPU 1, blocks 7-12 on GPU 2, blocks 13-18 on GPU 3, and blocks 19-24 on GPU 4.

---

## Chapter 9: Data Selection

### Figure 9.1: 

**Alt Text**: Line chart showing dataset size in tokens on y-axis from 10^10 to 10^14 versus year on x-axis from 2010 to 2030. Blue line shows training data growth with markers for models like GPT-2, GPT-3, and Chinchilla. Orange shaded region shows projected high-quality text exhaustion in the near term.

### Figure 9.2: The Optimization Triad

**Caption**: The Optimization Triad: Machine learning performance relies on three pillars: Algorithms (models), Systems (hardware/software), and Data Selection. While algorithms and systems have traditionally received the most attention, optimizing data selection (Input Optimization) offers a third, powerful lever for scaling performance.

**Alt Text**: A triangular diagram with three nodes: Algorithms, Systems, and Data Selection. Arrows connect all three, forming a cycle. Data Selection is highlighted.

### Figure 9.3: The Data Selection Pipeline

**Caption**: The Data Selection Pipeline: A structured approach to increasing data value. Raw data is first pruned to remove redundancy (Static Pruning), then dynamically selected during training (Active Learning), and finally augmented to increase diversity (Synthesis). Each stage increases the Information-Compute Ratio (ICR).

**Alt Text**: A flow diagram showing the progression of data: Raw Data -> Static Pruning -> Dynamic Selection -> Synthetic Generation -> High Value Model. Arrows indicate the flow.

### Figure 9.4: Coreset Selection Strategy

**Caption**: Coreset Selection Strategy: Random sampling (left) selects uniformly, wasting budget on easy samples far from the decision boundary. Coreset selection (right) prioritizes samples near the boundary where the model is uncertain, capturing more information per sample.

**Alt Text**: Two scatter plots with a diagonal decision boundary. Left plot shows random dots selected. Right plot highlights dots near the boundary as selected.

### Figure 9.5: Active Learning Loop

**Caption**: Active Learning Loop: Instead of labeling all data, the model selects the most 'confusing' or informative samples from an unlabeled pool. These samples are sent to an Oracle (human annotator) and added to the training set. The model is retrained, and the cycle repeats, creating a feedback loop that maximizes data selection.

**Alt Text**: A cycle diagram: Unlabeled Pool -> Selection Strategy -> Oracle -> Labeled Set -> Model Training -> back to Selection Strategy.

### Figure 9.6: Cost Amortization in Foundation Models

**Caption**: Cost Amortization in Foundation Models: Training from scratch (left) requires full cost for each task. The foundation model approach (right) pays a large pre-training cost once, then amortizes it across many low-cost fine-tuning tasks. After 3–4 tasks, the foundation model approach becomes more cost-effective; after 10+ tasks, the savings are dramatic.

**Alt Text**: Two bar charts side by side. Left shows 10 equally tall bars representing train-from-scratch costs. Right shows one tall pre-training bar followed by 10 short fine-tuning bars, with total height much lower.

### Figure 9.7: The Domain Gap Problem

**Caption**: The Domain Gap Problem: Synthetic data (blue) and real data (orange) have different distributions. A model trained on synthetic data alone learns a boundary that fails on real data. Domain adaptation techniques aim to align these distributions or learn domain-invariant features.

**Alt Text**: Two overlapping bell curves representing synthetic and real data distributions, with a decision boundary that works for synthetic but misses real data.

### Figure 9.8: Data Selection Technique Selection Tree

**Caption**: Data Selection Technique Selection Tree: Start at the top by identifying your primary bottleneck, then follow the branches to find the most appropriate technique. Leaf nodes show recommended methods. Multiple paths may apply—combine techniques as needed.

**Alt Text**: A decision tree flowchart with diamond decision nodes and rectangular technique recommendations. Starts with bottleneck identification and branches to specific techniques.

### Figure 9.9: The Selection Inequality

**Caption**: The Selection Inequality: Data selection only improves end-to-end efficiency if the overhead of selection plus training on the subset is less than training on the full dataset. A lightweight selection function (proxy model, cached embeddings) keeps selection overhead low; an expensive selection function (full model forward pass) can negate the savings.

**Alt Text**: Bar chart comparing two approaches: baseline shows one tall bar for full training; data-efficient shows two shorter bars for selection overhead and subset training, with total shorter than baseline.

### Figure 9.10: The Data Roofline Model

**Caption**: The Data Roofline Model: Analogous to the compute Roofline, this diagnostic tool shows two regimes. Below the diagonal (data-bound), adding more data improves performance—invest in data collection. Above the diagonal (compute-bound), more data won't help without more training compute—invest in GPUs. The optimal operating point is at the knee where data and compute are balanced. Data selection techniques move you along the diagonal by extracting more value per sample.

**Alt Text**: A log-log plot with Data Quality on x-axis and Model Performance on y-axis. A diagonal line separates data-bound (lower) and compute-bound (upper) regimes. Points show system positions.

### Figure 9.11: Diminishing Returns of Data

**Caption**: Diminishing Returns of Data: Random sampling (gray) versus data-efficient selection (blue). The efficient strategy achieves higher performance with less data, reaching the convergence plateau much earlier. The red arrow shows the efficiency gap at a fixed dataset size.

**Alt Text**: A plot with X-axis 'Dataset Size' and Y-axis 'Performance'. Two curves start at 0. The 'Random' curve rises slowly. The 'Efficient' curve rises steeply and plateaus early.

---

## Chapter 10: Model Compression

### Figure 10.1: Optimization Stack

**Caption**: Optimization Stack: Model optimization progresses through three layers: efficient model representation, efficient numerics representation, and efficient hardware implementation.

**Alt Text**: Three stacked rectangular boxes labeled from top to bottom: Efficient Model Representation, Efficient Numerics Representation, Efficient Hardware Implementation. A vertical arrow spans the stack with More software at top and More hardware at bottom.

### Figure 10.2: Sparse Matrix Transformation

**Caption**: Sparse Matrix Transformation: Pruning removes small-magnitude weights (shown as white/zero in the right matrix) while preserving large-magnitude weights (shown in color), creating a sparse representation that reduces memory usage while maintaining model accuracy.

**Alt Text**: Two 11x11 matrices side by side. Left matrix shows dense weights with colored cells indicating magnitudes. Right matrix shows sparse version with most cells white (zero) and only high-magnitude values retained in color.

### Figure 10.3: Pruning Strategies

**Caption**: Pruning Strategies: Channel pruning adjusts filter sizes within layers, while layer pruning removes entire layers and necessitates reconnection of remaining network components. These approaches reduce model size and computational cost, but require fine-tuning to mitigate performance loss due to reduced model capacity.

**Alt Text**: Side-by-side diagrams showing channel pruning (left) and layer pruning (right). Each shows three-stage CNN with feature maps as 3D blocks connected by dashed lines. Red highlights indicate pruned channels or layers.

### Figure 10.4: Pruning Strategies

**Caption**: Pruning Strategies: Unstructured pruning achieves sparsity by removing individual weights, requiring specialized hardware for efficient computation, while structured pruning removes entire neurons or filters, preserving network structure and enabling acceleration on standard hardware. This figure contrasts the resulting weight matrices and network architectures from both approaches, highlighting the trade-offs between sparsity level and computational efficiency. Source: [@qi2021efficient].

**Alt Text**: Three-panel diagram. Left shows unstructured pruning with dashed connections in a neural network. Middle and right show structured pruning: fully connected network with pruned neurons and CNN with pruned filters shown as dashed squares.

### Figure 10.5: Iterative Pruning Performance

**Caption**: Iterative Pruning Performance: Gradual channel removal with interleaved fine-tuning maintains high accuracy while reducing model size. This figure shows a 0.4% accuracy drop with a 27% reduction in channels, demonstrating the benefits of distributing structural modifications across multiple iterations. This approach contrasts with one-shot pruning, which often leads to significant performance degradation.

**Alt Text**: Three-row workflow showing iterative pruning. Each row displays CNN architecture, prune step with red arrow, accuracy drop box, fine-tune gears icon, and accuracy recovery. Values progress from 0.995 to 0.991 final accuracy.

### Figure 10.6: One-Shot Pruning Impact

**Caption**: One-Shot Pruning Impact: Aggressive removal of architectural components, like the 27% of channels shown, causes significant initial accuracy loss because the network struggles to adapt to significant structural changes simultaneously. Fine-tuning partially recovers performance, but establishes that iterative pruning preserves accuracy more effectively than single-step approaches.

**Alt Text**: Single-row workflow showing one-shot pruning. CNN with six red-highlighted channels to prune, followed by accuracy drop from 0.995 to 0.914, fine-tuning gears, and partial recovery to 0.943.

### Figure 10.7: Winning Ticket Discovery

**Caption**: Winning Ticket Discovery: Iterative pruning and weight resetting identify subnetworks within larger models that, when trained in isolation, achieve comparable or superior accuracy, challenging the conventional view of pruning as solely a compression technique. This process establishes that well-initialized subnetworks exist and can be efficiently trained, suggesting that much of a large network's capacity may be redundant.

**Alt Text**: Cyclic flowchart with four stages: dense network, train to convergence, prune smallest weights, reset remaining weights to initial values. Arrows form iterative loop that progressively identifies winning ticket subnetwork.

### Figure 10.8: Soft Target Distribution

**Caption**: Soft Target Distribution: The teacher's relative confidence levels indicate which classes are semantically similar (e.g., cat vs. dog), providing a much richer supervision signal than a binary "correct" label.

**Alt Text**: Bar chart showing probability distribution across three animal classes: Cat at 85 percent, Dog at 10 percent, Fox at 5 percent. Demonstrates how soft labels capture inter-class similarity.

### Figure 10.9: Knowledge Distillation Workflow

**Caption**: Knowledge Distillation Workflow: The student model inherits the teacher's generalization capabilities by matching its softened probability distributions. This dual-loss training enables the student to absorb complex class relationships that are absent from binary labels.

**Alt Text**: Block diagram showing knowledge distillation. Input flows to both teacher and student models. Teacher outputs soft labels via temperature-scaled softmax. Student outputs feed into distillation loss and student loss functions.

### Figure 10.10: Low-Rank Factorization

**Caption**: Low-Rank Factorization: Decomposing a matrix into lower-rank approximations reduces the number of parameters needed for storage and computation, enabling efficient model representation. By expressing a matrix $a$ as the product of two smaller matrices, $u$ and $v$, we transition from storing $m \times n$ parameters to $m \times k + k \times n$ parameters, with $k$ representing the reduced rank. Source: The Clever Machine.

**Alt Text**: Three rectangular boxes showing matrix factorization. Large M matrix of size m by n approximately equals product of narrower L matrix of size m by k and wider R-transpose matrix of size k by n.

### Figure 10.11: Tensor Decomposition

**Caption**: Tensor Decomposition: Multi-dimensional tensors enable compact representations of high-dimensional data by factorizing them into lower-rank components, reducing computational costs and memory requirements compared to direct manipulation of the original tensor. This technique extends matrix factorization to handle the multi-way interactions common in modern machine learning models like convolutional neural networks. Source: [@xinyu].

**Alt Text**: 3D tensor cube with dimensions M, N, T decomposed into sum of three factor matrices U, V, W of reduced dimensions. Small highlighted element shows how single tensor entry decomposes into factor products.

### Figure 10.12: Neural Architecture Search Flow

**Caption**: Neural Architecture Search Flow: Automated NAS techniques iteratively refine model architectures and their weights, jointly optimizing for performance and efficiency, a departure from manual design approaches that rely on human expertise and extensive trial-and-error. This process enables the discovery of novel, high-performing architectures tailored to specific computational constraints.

**Alt Text**: Three-box flowchart showing NAS process. Search Space box feeds into Search Strategy box, which exchanges Architecture and Performance estimate with Performance Estimation Strategy box in a feedback loop.

### Figure 10.13: Energy Costs

**Caption**: Energy Costs: Lower precision reduces computational energy, illustrating trade-offs in model accuracy. Machine learning systems can optimize efficiency by reducing floating-point operations from 32-bit to 16-bit or even lower for significant savings. Source: IEEE spectrum.

**Alt Text**: Bar chart comparing energy consumption in picojoules for arithmetic operations and memory accesses. FP32 multiply uses 3.7 pJ, INT8 add uses 0.03 pJ. SRAM reads range from 5 to 50 pJ depending on cache size.

### Figure 10.14: Quantization Impact

**Caption**: Quantization Impact: Moving from FP32 to INT8 reduces inference time by up to 4 times while decreasing model size by a factor of 4, making models more efficient for resource-constrained environments.

**Alt Text**: Two stacked bar charts comparing FP32 and INT8. Left chart shows inference time in milliseconds for Inception, MobileNet, and ResNet. Right chart shows model size in megabytes. INT8 consistently smaller and faster.

### Figure 10.15: 

**Alt Text**: Histogram showing quantization error distribution weighted by probability density. Bell-shaped curve centered near zero with tails extending to positive and negative errors, illustrating typical quantization noise pattern.

### Figure 10.16: Floating-Point Precision

**Caption**: Floating-Point Precision: Reduced-precision formats like FP16 and bfloat16 trade off numerical range for computational efficiency and memory savings. Bfloat16 maintains the exponent size of FP32, preserving its dynamic range and suitability for training, while FP16's smaller exponent limits its use to inference or carefully scaled training scenarios.

**Alt Text**: Three horizontal bit-layout diagrams. FP32 shows 1-bit sign, 8-bit exponent, 23-bit mantissa. FP16 shows 1-bit sign, 5-bit exponent, 10-bit mantissa. BFloat16 shows 1-bit sign, 8-bit exponent, 7-bit mantissa.

### Figure 10.17: Quantization Complexity Roadmap

**Caption**: Quantization Complexity Roadmap: Three progressive tiers of quantization techniques, from foundational approaches suitable for quick deployment to research frontier methods for extreme resource constraints, reflecting increasing implementation effort, resource requirements, and potential accuracy trade-offs.

**Alt Text**: Tiered diagram with three levels. Foundation tier includes PTQ and basic INT8. Production tier adds QAT and mixed precision. Research frontier tier shows INT4, binary, and ternary quantization with icons for increasing complexity.

### Figure 10.18: Post-Training Quantization

**Caption**: Post-Training Quantization: Calibration with a representative dataset determines optimal quantization ranges for model weights and activations, minimizing information loss during quantization to create efficient, lower-precision models. This process converts a pre-trained model into a quantized version suitable for deployment on resource-constrained devices.

**Alt Text**: Vertical flowchart with four boxes connected by arrows. Pre-trained model and Calibration data feed into Calibration step, which feeds into Quantization step, producing final Quantized model output.

### Figure 10.19: 

**Alt Text**: Histogram of ResNet50 activation values showing right-skewed distribution. Most values cluster near zero with long tail extending to outliers around 2.1, demonstrating challenge for quantization range selection.

### Figure 10.20: Calibration Range Selection

**Caption**: Calibration Range Selection: Symmetric calibration uses a fixed range around zero, while asymmetric calibration adapts the range to the data distribution, potentially minimizing quantization error and preserving model accuracy. Choosing an appropriate calibration strategy balances precision with the risk of saturation for outlier values.

**Alt Text**: Two side-by-side mapping diagrams. Left shows symmetric calibration with range from -1 to 1 mapping to -127 to 127 with zero aligned. Right shows asymmetric calibration with range -0.5 to 1.5 mapping with shifted zero point.

### Figure 10.21: Quantization Range Variation

**Caption**: Quantization Range Variation: Different convolutional filters exhibit unique activation ranges, necessitating per-filter quantization to minimize accuracy loss during quantization. Adjusting the granularity of clipping ranges—as shown by the differing scales for each filter—optimizes the trade-off between model size and performance. Source: [@gholami2021survey].

**Alt Text**: Four rows showing CNN filters with Gaussian weight distributions. Each filter has different clipping ranges shown as red and blue dashed lines. Layer-wise clipping uses same range; channel-wise uses per-filter ranges.

### Figure 10.22: Quantization and Weight Precision

**Caption**: Quantization and Weight Precision: Reducing weight and activation precision from float32 to INT8 significantly lowers model size and computational cost during inference by representing values with fewer bits, though it may introduce a trade-off with model accuracy. This process alters the numerical representation of model parameters and intermediate results, impacting both memory usage and processing speed. Source: HarvardX.

**Alt Text**: Matrix multiplication diagram with three steps. Blue squares show input activations, red squares show quantized weights, and green squares show output activations. Arrows indicate computation flow through multiply-accumulate operations.

### Figure 10.23: Quantization-Aware Training

**Caption**: Quantization-Aware Training: Retraining a pre-trained model with simulated low-precision arithmetic adapts weights to mitigate accuracy loss during deployment with reduced numerical precision, enabling efficient inference on resource-constrained devices. This process refines the model to become robust to the effects of quantization, maintaining performance despite lower precision representations.

**Alt Text**: Vertical flowchart showing QAT process. Pre-trained model feeds into Quantization step, then Retraining/Finetuning step with Training data input, producing final Quantized model output.

### Figure 10.24: Hybrid Quantization Approach

**Caption**: Hybrid Quantization Approach: Post-training quantization (PTQ) generates an initial quantized model that serves as a warm start for quantization-aware training (QAT), accelerating convergence and mitigating accuracy loss compared to quantizing a randomly initialized network. This two-stage process leverages the efficiency of PTQ while refining the model with training data to optimize performance under low-precision constraints.

**Alt Text**: Vertical flowchart with two grouped stages. PTQ stage shows pretrained model through quantize and calibrate steps. QAT stage shows fine-tuning step. Calibrate data feeds PTQ; Training data feeds QAT.

### Figure 10.25: Early Exit Architecture

**Caption**: Early Exit Architecture: Transformer layers dynamically adjust computation by classifying each layer's output and enabling early termination if sufficient confidence is reached, reducing latency and power consumption for resource-constrained devices. This approach allows for parallel evaluation of different exit paths, improving throughput on hardware accelerators like GPUs and TPUs. Source: [@xin-etal-2021-berxit].

**Alt Text**: Flowchart with input feeding n transformer layers in sequence. Each layer connects to a classifier, confidence estimator, and exit point. Arrows show continue paths for low confidence.

### Figure 10.26: Conditional Computation

**Caption**: Conditional Computation: Switch transformers enhance efficiency by dynamically routing tokens to specialized expert subnetworks, enabling parallel processing and reducing the computational load per input. This architecture implements a form of mixture of experts where a gating network selects which experts process each token, allowing for increased model capacity without a proportional increase in computation. *Source: [@fedus2021switch]*.

**Alt Text**: Two-part diagram. Left shows Switch Transformer block with self-attention, add-normalize, switching FFN layer, and add-normalize. Right shows expanded view with router selecting one of four FFN experts per token based on probability.

### Figure 10.27: Block Sparse Representation

**Caption**: Block Sparse Representation: NVIDIA's cusparse library efficiently stores block sparse matrices by exploiting dense submatrix structures, enabling accelerated matrix operations while maintaining compatibility with dense matrix computations through block indexing. This approach reduces memory footprint and arithmetic complexity for sparse linear algebra, important for scaling machine learning models. _Source: NVIDIA._

**Alt Text**: Grid of 3x3 matrix blocks with varying shades indicating dense submatrices. Adjacent index array shows non-zero block positions. Gray blocks represent zeros, colored blocks represent dense submatrices stored separately.

### Figure 10.28: Sparse Matrix Multiplication

**Caption**: Sparse Matrix Multiplication: Block sparsity optimizes matrix operations by storing only non-zero elements and using structured indexing, enabling efficient GPU acceleration for neural network computations. This technique maintains compatibility with dense matrix operations while reducing memory access and computational cost, particularly beneficial for large-scale models. Source: PyTorch blog [@pytorch_sparsity_blog].

**Alt Text**: Side-by-side comparison of dense and 2:4 sparse GEMM on Tensor Cores. Left shows 8-element row multiplication. Right shows 4-element sparse row with 2-bit indices selecting matching elements from dense B matrix.

### Figure 10.29: Compression Trade-Offs

**Caption**: Compression Trade-Offs: Combining pruning and quantization achieves superior compression ratios with minimal accuracy loss compared to quantization or singular value decomposition (SVD) alone, demonstrating the impact of different numerical precision optimization techniques on model size and performance. Architectural and numerical optimizations can complement each other to efficiently deploy machine learning models via this figure. Source: [@han2015deep].

**Alt Text**: Line graph of accuracy loss versus model size ratio. Four curves show pruning plus quantization achieving smallest size at near-zero loss, followed by pruning only, quantization only, and SVD requiring largest size to maintain accuracy.

### Figure 10.30: AutoML Workflow

**Caption**: AutoML Workflow: Automated machine learning (automl) streamlines model development by structurally automating data preprocessing, model selection, and hyperparameter tuning, contrasting with traditional workflows requiring extensive manual effort for each stage. This automation enables practitioners to define high-level objectives and constraints, allowing automl systems to efficiently explore a vast design space and identify optimal model configurations.

**Alt Text**: Two circular workflow diagrams side by side. Left shows traditional ML with five manual steps. Right shows AutoML with three steps where preprocessing, training, and evaluation are automated in a single AutoML node.

### Figure 10.31: 

**Alt Text**: Grid of 96 small color images showing AlexNet first-layer convolutional kernels. Patterns include oriented edges, color blobs, and Gabor-like filters learned from ImageNet training.

### Figure 10.32: 

**Alt Text**: Heatmap visualization of a pruned neural network with weight matrix blocks. Darker regions indicate higher sparsity where more weights have been removed. Lighter regions show retained weights.

---

## Chapter 11: AI Acceleration

### Figure 11.1: Hardware Specialization Trajectory

**Caption**: Hardware Specialization Trajectory: Computing architectures progressively incorporate specialized accelerators to address emerging performance bottlenecks and workload demands, mirroring a historical pattern from floating-point units to graphics processors and, ultimately, machine learning accelerators. This evolution reflects a strategy for improving computational efficiency by tailoring hardware to specific task characteristics and advancing increasingly complex applications.

**Alt Text**: Timeline spanning 1980s to 2020s showing hardware evolution: floating-point units, GPUs with hardware transform and lighting, media codecs, TPUs with tensor cores, and application-specific AI engines.

### Figure 11.2: Anatomy of a Modern AI Accelerator

**Caption**: Anatomy of a Modern AI Accelerator: AI accelerators integrate specialized processing elements containing tensor cores, vector units, and special function units, supported by a hierarchical memory system from high-bandwidth memory down to local caches. This architecture maximizes data reuse and parallel execution while minimizing energy-intensive data movement, forming the foundation for 100-1000× performance improvements over general-purpose processors.

**Alt Text**: Block diagram showing AI accelerator architecture: CPU connects to DRAM stacks and processing element grid containing tensor cores, vector units, and local caches in hierarchical arrangement.

### Figure 11.3: 

**Alt Text**: Line graph of NVIDIA GPU INT8 performance from 2012 to 2023 showing exponential growth from K20X at 4 TOPS to H100 at 4000 TOPS, a 1000x increase over the decade.

### Figure 11.4: Systolic Array Dataflow

**Caption**: Systolic Array Dataflow: Processing elements within the array execute matrix operations by streaming data in a pipelined manner, maximizing operand reuse and minimizing memory access compared to traditional memory-compute architectures. This spatial and temporal locality enables efficient parallel computation, as exemplified by the multiply-accumulate units in Google's tpuv4.

**Alt Text**: Systolic array diagram with control unit feeding data streams into processing element grid. Elements perform multiply-accumulate operations with results flowing through accumulator chain.

### Figure 11.5: Host-Accelerator Data Transfer

**Caption**: Host-Accelerator Data Transfer: AI workloads require frequent data movement between CPU memory and accelerators; this figure details the sequential steps of copying input data, executing computation, and transferring results, each introducing potential performance bottlenecks. Understanding this data transfer sequence helps optimize AI system performance and minimize latency.

**Alt Text**: Four-step data flow diagram: (1) copy data from main memory to GPU memory, (2) CPU instructs GPU, (3) GPU executes in parallel, (4) results copy back to main memory.

### Figure 11.6: Matrix Tiling

**Caption**: Matrix Tiling: Partitioning large matrices into smaller tiles optimizes data reuse and reduces memory access overhead during computation. This technique improves performance on AI accelerators by enabling efficient loading and processing of data in fast memory, minimizing transfers from slower main memory.

**Alt Text**: Three matrices A, B, C with highlighted tiles showing how matrix multiplication partitions into smaller blocks. Dimensions labeled M, N, K with corresponding tile sizes Mtile, Ntile, Ktile.

---

## Chapter 12: Benchmarking AI

### Figure 12.1: ImageNet Benchmark

**Caption**: ImageNet Benchmark: Advancements in GPU technology have driven improvements in ImageNet classification accuracy since 2012, showcasing the interplay between hardware and algorithmic progress.

**Alt Text**: Dual-axis chart with blue line showing top-5 error rate declining from 28% to 7% and green bars showing GPU entries rising from 0 to 110 between 2010 and 2014.

### Figure 12.2: Benchmarking Granularity

**Caption**: Benchmarking Granularity: ML system performance assessment occurs at multiple levels, from end-to-end application metrics to individual model and hardware component efficiency, enabling targeted optimization and bottleneck identification. This hierarchical approach allows practitioners to systematically analyze system performance and prioritize improvements based on specific component limitations.

**Alt Text**: Block diagram showing three evaluation layers: neural network nodes on left, model components in center, and end-to-end application with compute nodes on right, connected by dashed lines.

### Figure 12.3: Benchmark Granularity Trade-offs

**Caption**: Benchmark Granularity Trade-offs: The core trade-off in benchmarking granularity between isolation/diagnostic power and real-world representativeness. Micro-benchmarks provide high diagnostic precision but limited real-world relevance, while end-to-end benchmarks capture realistic system behavior but offer less precise component-level insights. Effective ML system evaluation requires strategic combination of all three levels.

**Alt Text**: Scatter plot with three labeled points along diagonal: micro-benchmarks at high isolation, macro-benchmarks at medium, and end-to-end benchmarks at high representativeness.

### Figure 12.4: Benchmark Workflow

**Caption**: Benchmark Workflow: AI benchmarks standardize evaluation through a structured pipeline, enabling reproducible performance comparisons across different models and systems. This workflow systematically assesses AI capabilities by defining tasks, selecting datasets, training models, and rigorously evaluating results.

**Alt Text**: Workflow diagram showing nine stages from problem definition through deployment, with detailed views of anomaly detection system, model training, quantization, and ARM embedded implementation.

### Figure 12.5: MLPerf Training Progress

**Caption**: MLPerf Training Progress: Standardized benchmarks reveal that machine learning training performance consistently surpasses moore's law, indicating substantial gains from systems-level optimizations. These trends emphasize how focused measurement and iterative improvement drive rapid advancements in ML training efficiency and scalability. Source: [@tschand2024mlperf].

**Alt Text**: Line chart with nine model benchmarks from 2018 to 2024 showing relative performance gains up to 48x for Mask R-CNN, all exceeding the Moore's Law baseline of 6.6x.

### Figure 12.6: Power Measurement Boundaries

**Caption**: Power Measurement Boundaries: MLPerf defines system boundaries for power measurement, ranging from single-chip devices to full data center nodes, to enable fair comparisons of energy efficiency across diverse hardware platforms. These boundaries delineate which components' power consumption is included in reported metrics, impacting the interpretation of performance results. Source: [@tschand2024mlperf].

**Alt Text**: System diagram showing four measurement boundaries: Tiny SoC with compute units, Inference SoC with accelerators and DRAM, Inference Node with cooling and NIC, and Training Rack with compute nodes.

### Figure 12.7: Energy Efficiency Gains

**Caption**: Energy Efficiency Gains: Successive MLPerf inference benchmark versions consistently improve energy efficiency (samples/watt) across diverse system scales (datacenter, edge, and tiny), reflecting ongoing advancements in both hardware and software optimization for AI workloads. Standardized measurement protocols enable meaningful comparisons of energy efficiency improvements across different AI systems and deployment scenarios, driving sector-wide progress toward sustainable AI technologies. Source: [@tschand2024mlperf].

**Alt Text**: Three line charts showing normalized energy efficiency across MLPerf versions: datacenter models up to 378x gain, edge models up to 4x, and tiny models up to 1070x improvement.

### Figure 12.8: Hardware-Dependent Accuracy

**Caption**: Hardware-Dependent Accuracy: Model performance varies significantly across hardware platforms, indicating that architectural efficiency is not solely determined by design but also by hardware compatibility. Multi-hardware models exhibit comparable accuracy to mobilenetv3 large on CPU and GPU configurations, yet achieve substantial gains on EdgeTPU and DSP, emphasizing the importance of hardware-aware model optimization for specialized computing environments. Source: [@chu2021discovering].

**Alt Text**: Five scatter plots comparing model accuracy versus latency across CPU, GPU, EdgeTPU, and DSP platforms, with arrow showing MobileNetV3 gaining on EdgeTPU and DSP versus CPU and GPU.

### Figure 12.9: Performance Spectrum

**Caption**: Performance Spectrum: Scientific applications and edge devices demand vastly different computational resources, spanning multiple orders of magnitude in data rates and latency requirements. Consequently, traditional benchmarks focused solely on accuracy are insufficient; specialized evaluation metrics and benchmarks like MLPerf become essential for optimizing AI systems across diverse deployment scenarios. Source: [@duarte2022fastml].

**Alt Text**: Log-scale scatter plot of data rate versus computation time, showing scientific applications from LHC sensors at 10^14 B/s and nanoseconds to mobile devices at 10^4 B/s and seconds.

### Figure 12.10: Development Paradigms

**Caption**: Development Paradigms: Model-centric AI prioritizes architectural innovation with fixed datasets, while data-centric AI systematically improves dataset quality (annotations, diversity, and bias) with consistent model architectures to achieve performance gains. Modern research indicates that strategic data enhancement often yields greater improvements than solely refining model complexity.

**Alt Text**: Side-by-side diagrams: model-centric AI shows data cylinders feeding CPU with feedback loop to model, data-centric AI shows feedback loop to data instead. Double arrow indicates complementary approaches.

### Figure 12.11: Dataset Saturation

**Caption**: Dataset Saturation: AI systems surpass human performance on benchmark datasets, indicating that continued gains may not reflect genuine improvements in intelligence but rather optimization to fixed evaluation sets. This trend underscores the need for dynamic, challenging datasets that accurately assess AI capabilities and drive meaningful progress beyond simple pattern recognition. Source: [@kiela2021dynabench].

**Alt Text**: Line chart showing five AI capabilities crossing human performance baseline from 1998 to 2020: handwriting, speech, image recognition, reading comprehension, and language understanding.

---

## Chapter 13: Model Serving Systems

### Figure 13.1: The Inference Pipeline

**Caption**: The Inference Pipeline: ML serving systems transform raw inputs into final outputs through sequential stages—preprocessing, neural network computation, and postprocessing. The neural network represents just one component; preprocessing and postprocessing rely on traditional computing and often dominate total latency in optimized systems.

**Alt Text**: Flow diagram showing six connected boxes: Raw Input, Preprocessing, Neural Network, Raw Output, Postprocessing, Final Output. Preprocessing and postprocessing are labeled Traditional Computing; neural network is labeled Deep Learning.

### Figure 13.2: Inference Server Anatomy

**Caption**: Inference Server Anatomy: Modern inference servers organize request processing into a decoupled pipeline. The Network Ingress handles high-concurrency protocols (HTTP/gRPC), the Queue buffers bursts of traffic, and the Dynamic Batcher aggregates individual requests into optimized tensors. The Inference Runner manages the low-level execution on the hardware accelerator, ensuring the GPU remains utilized through asynchronous execution.

**Alt Text**: Flowchart showing 6-stage inference server pipeline: Client to Network Ingress to Request Queue (cylinder) to Dynamic Batcher, then down to Inference Runner to Accelerator. Arrows connect stages sequentially.

### Figure 13.3: Request Pipelining

**Caption**: Request Pipelining: Pipelining hides latency by overlapping independent operations across different hardware resources. In pipelined execution (B), the CPU processes the next request's data while the GPU executes the current request's inference. This increases the GPU duty cycle toward 100%, effectively doubling or tripling throughput on the same hardware without changing the model.

**Alt Text**: Two timing diagrams. A (Serial): alternating CPU preprocessing, GPU inference, and idle blocks in sequence. B (Pipelined): two parallel rows where CPU preprocessing overlaps with GPU inference, eliminating idle time.

---

## Chapter 14: Machine Learning Operations (MLOps)

### Figure 14.1: MLOps Lifecycle

**Caption**: MLOps Lifecycle: MLOps extends DevOps principles to manage the unique challenges of machine learning systems, including data versioning, model retraining, and continuous monitoring. This diagram outlines the iterative workflow encompassing data engineering, model development, and reliable deployment for sustained performance in production.

**Alt Text**: Infinity-loop diagram with three phases. Design phase: requirements, use-case prioritization, data availability. Model Development: data engineering, model engineering, testing. Operations: deployment, CI/CD pipeline, monitoring and triggering.

### Figure 14.2: ML System Complexity

**Caption**: ML System Complexity: Most engineering effort in a typical machine learning system concentrates on components surrounding the model itself: data collection, feature engineering, and system configuration rather than the model code. This distribution reveals the operational challenges and potential for technical debt arising from these often-overlooked areas of an ML system. Source: [@sculley2015hidden].

**Alt Text**: Hub-and-spoke diagram with ML system at center. Ten surrounding components connected by arrows: data collection, verification, feature extraction, configuration, resource management, serving infrastructure, monitoring, analysis tools, and ML code.

### Figure 14.3: ML Technical Debt Taxonomy

**Caption**: ML Technical Debt Taxonomy: Machine learning systems accumulate distinct forms of technical debt that emerge from data dependencies, model interactions, and evolving requirements. This hub-and-spoke diagram illustrates the primary debt patterns: boundary erosion undermines modularity, correction cascades propagate fixes through dependencies, feedback loops create hidden coupling, while data, configuration, and pipeline debt reflect poorly managed artifacts and workflows. Understanding these patterns enables systematic engineering approaches to debt prevention and mitigation.

**Alt Text**: Hub-and-spoke diagram with Hidden Technical Debt at center. Six debt categories radiate outward: Configuration Debt, Feedback Loops, Data Debt, Pipeline Debt, Correction Cascades, and Boundary Erosion, each annotated with specific failure patterns.

### Figure 14.4: Correction Cascades

**Caption**: Correction Cascades: Iterative refinements in ML systems often trigger dependent fixes across the workflow, propagating from initial adjustments through data, model, and deployment stages. Color-coded arcs represent corrective actions stemming from sources of instability, while red arrows and the dotted line indicate escalating revisions, potentially requiring a full system restart.

**Alt Text**: Timeline diagram with seven ML stages from problem statement to deployment. Color-coded arcs show correction cascades: red for domain expertise gaps, blue for real-world brittleness, orange for poor documentation. Dashed arrows indicate restarts.

### Figure 14.5: MLOps Stack Layers

**Caption**: MLOps Stack Layers: Modular architecture organizes machine learning system components, from model development and orchestration to infrastructure, facilitating automation, reproducibility, and scalable deployment. Each layer builds upon the one below, enabling cross-team collaboration and supporting the entire ML lifecycle from initial experimentation to long-term production maintenance.

**Alt Text**: Layered architecture diagram. Top row: ML Models, Frameworks, Orchestration, Infrastructure, Hardware. MLOps section spans orchestration tasks (data management through model serving) and infrastructure tasks (job scheduling through monitoring).

### Figure 14.6: ML CI/CD Pipeline

**Caption**: ML CI/CD Pipeline: Automated workflows streamline model development by integrating version control, testing, and deployment, enabling continuous delivery of updated models to production. This pipeline emphasizes data and model validation, automated retraining triggers, and model registration with metadata for reproducibility and governance. Source: HarvardX.

**Alt Text**: Pipeline diagram showing continuous training workflow. Central box contains data validation, transformation, training, evaluation, and registration stages. Three repositories connect: dataset and feature, metadata and artifact, model.

### Figure 14.7: Data Drift Impact

**Caption**: Data Drift Impact: Declining model performance over time results from data drift, where the characteristics of production data diverge from the training dataset. Monitoring key metrics longitudinally allows MLOps engineers to detect this drift and trigger model retraining or data pipeline adjustments to maintain accuracy.

**Alt Text**: Three-panel visualization over time. Top: incoming data samples coded green or orange. Middle: feature distribution shifting from online to offline sales channel. Bottom: line graph showing model accuracy declining as distribution shifts increase.

### Figure 14.8: Uptime Dependency Stack

**Caption**: Uptime Dependency Stack: Robust ML service uptime relies on monitoring a layered stack of interdependent components, from infrastructure to model performance, mirroring the complexity of modern software systems. Operational maturity necessitates observing this entire stack to proactively address potential failures and maintain service levels under varying conditions.

**Alt Text**: Iceberg diagram with uptime visible above waterline. Hidden below: model accuracy, data drift, concept drift, broken pipelines, schema changes, model bias, data outages, underperforming segments. Labels indicate data, model, and service health.

### Figure 14.9: ClinAIOps Feedback Loops

**Caption**: ClinAIOps Feedback Loops: The cyclical framework coordinates data flow between patients, clinicians, and AI systems to support continuous model improvement and safe clinical integration. These interconnected loops enable iterative refinement of AI models based on real-world performance and clinical feedback, fostering trust and accountability in healthcare applications. Source: [@chen2023framework].

**Alt Text**: Circular diagram with three nodes: patient, clinician, and AI system. Arrows form cyclic flow: patient provides monitoring data, clinician sets therapy regimen, AI generates alerts and recommendations. Inner and outer loops show feedback pathways.

### Figure 14.10: Patient-Clinician Interaction

**Caption**: Patient-Clinician Interaction: Continuous monitoring data informs collaborative discussions between patients and clinicians, shifting focus from data collection to actionable insights for lifestyle modifications and improved health management. This loop prioritizes patient engagement and contextual understanding to facilitate personalized care beyond traditional clinical visits. Source: [@chen2023framework].

**Alt Text**: Three-panel diagram showing ClinAIOps loops. Patient-AI loop: patient monitors blood pressure, AI recommends titrations. Clinician-AI loop: clinician sets limits, AI sends alerts. Patient-clinician loop: both discuss therapy trends and modifiers.

---

## Chapter 15: Responsible Engineering

### Figure 15.1: 

**Alt Text**: Nested oval diagram showing governance layers from innermost to outermost: Team (reliable systems, software engineering), Organization (safety culture, organizational design), Industry (trustworthy certification, external reviews), and Government Regulation.

### Figure 15.2: 

**Alt Text**: Diagram showing two subgroups A and B with different score distributions. Vertical threshold lines at 75% and 81.25% show how the same threshold produces different approval rates for each group.

### Figure 15.3: 

**Alt Text**: Horizontal spectrum showing model types from more interpretable (decision trees, linear regression, logistic regression) to less interpretable (random forest, neural network, convolutional neural network).

---

_Total: 175 figures across 15 chapters_

# Data Representation Design Pattern 
#### How data should be processed by machine learning models?

**Content**:
- [The Goal of Learning](#the-goal-of-learning)
- [Data Representation](#data-representation)
- [Data Types](#data-types)
- [Simple Numerical Representation](#simple-numerical-representation)
- [Linear Scaling](#linear-scaling)
- [Outliers](#outliers)
- [Nonlinear Transformation](#nonlinear-transformation)
- [Parametric Transformation](#parametric-transformation)
- [Histogram Equalization](#histogram-equalization)
- [Categorical Inputs](#categorical-inputs)
- [Design #1: Hashed Design Pattern](#design-1-hashed-design-pattern)
- [Design #2: Embedding Design Pattern](#design-2-embedding-design-pattern)
- [Design #3: Feature Cross Design Pattern](#design-3-feature-cross-design-pattern)
- [Design #4: Multi-modal Input](#design-4-multi-modal-input)


# The Goal of Learning 
**Risk minimization**: Given the data points extracted from $x,y ~ p(x,y) $, we are willing to find the function $y=f_\theta(x)$, the loss function then would be $R(\theta)=E_{x,y~p(x,y)}(L(y, f_\theta(x)))$.
- We optimize the expectation as it gives us the **best generalization guarantee**.
- Optimizing for individual points would bias toward noise/outliers.

**Are There Alternatives?**
- **Robust/Minmax Risk**, $R(\theta)=\max_{x,y ~U} L(y, f_\theta(x)) $
- Qunatile / Median Risk 
- **Trimmed / Capped / Winsorized Risk**. $R(\theta)=E{x,y~p(x,y)}(L(y, f_\theta(x); w_k))$ 
- Multi-objective 

# Data Representation 
**Data representation** – Machine learning models are often represented as a mathematical function operates on input features. 
- Take input as real-world quantities and features and manipulated inputs. 
- Data representation design pattern states how input data should be transformed into features. 

**Common design patterns**: 
- [Hashed design pattern](#design-1-hashed-design-pattern)
- [Embedding design pattern](#design-2-embedding-design-pattern)
- [Feature cross design pattern](#feature-cross) 
- Multi-modal design pattern 

# Data Types 
**Structured Data** – Tabular, clearly organized into rows/columns, easy to query.
- **Numerical** – Continuous, Discrete
- **Categorical** – Nominal, Ordinal
- **Time-series (tabular form)** – Stock prices, sensor logs with timestamps

**Unstructured data** – Data without predefined schema, raw form, harder to process directly. 
- Text (documents, tweets, chat logs)
- Image (photos, MRI scans, satellite images)
- Audio (speech, music, environmental sound)
- Video (sequence of images + audio)

**Semi-Structured** – Graphs/Networks, XML/JSON, Sensor data fusion

# Simple Numerical Representation
**Numerical representation** – Numerical representation is an idiom for machine learning rather than a design pattern. 
- As it is stated, machine learning models accept numerical input (like neural networks, random forest, and support vector machines). If the inputs are numeric in common sense feed it into the model without changing it.
- It may require only scaling. Why? 
    - Speeding up the training (some algorithms like SGD, make large updates for some features). 
    - Reduce the algorithm sensitivity to the scale of features (like K-means).
    - Make regularization methods effective and uniform.

# Linear Scaling 
Linear scaling methods – Transform features linearly without changing the overall shape of the distribution.
- **Use case** – Suitable when data are well-structured and roughly symmetric.
- **Quote** – Linear scaling preserves relationships between values while making them compatible with algorithms sensitive to scale.

**Methods** 
- **Min-max scaler** – Linearly rescales data to a fixed range and sensitive to outliers 
- **Z-score normalization** – Centers data at mean 0 with standard deviation 1.
- **Rule of thumb**:
    - **Min-Max** → When you know bounds and data has few outliers.
        $$
            \hat x = \frac {x-\min(x)}{\max(x) - \min(x)} \in [0, 1] \\
            \hat x = 2\hat x - 1 \in [-1, 1]
        $$
    - **Z-Score** → When data is unbounded.
    $$
        \hat x = \frac {x-\mu}{\sigma} \in [-3, 3]
    $$
## Outliers 
**Outliers** – Data points that differ significantly from the majority of the data.
- They are unusually large or small compared to the rest of the observations.
- Outliers can occur in any type of data: continuous, categorical, or multivariate.
<!-- ![](./pic/Outliers.png) -->
**Are they bad data?**
- Measurement or recording errors (e.g., faulty sensors, typos).
- Natural variability in the data (e.g., extremely high incomes, rare events).
- Data processing issues (e.g., incorrect units, duplicates).

## Trimming
**Trimming** – Linear scaling (Min-Max, Z-score) assumes that extreme values do not dominate the scale. Trimming methods reduce the impact of extreme values before or during scaling.
- Preserving the shape of most data while limiting the effect of outliers.
    - **Clipping** – Cap the values to specific lower and upper boundaries.
    - **Winsorizing** – Replace extreme values with the nearest value within chosen percentiles. 
- It keeps the distribution shape of many data points intact

# Nonlinear Transformation

**Skewed distribution** – where the data is not symmetrically distributed around the mean, it will: 
- Violate normal assumption (residual is normally distributed) 
- Reducing the learning efficiency 
- Influences distance based algorithms
- Wrong impact on feature importance and scaling

As it discussed, the linear transformation will not change the distribution and they are effective if the underlying distribution of the data is symmetric. **What if the this assumption is not hold?** <br/> 
- As we saw, the learning goal is founded around optimizing the expectation values. **Is it a good choice for skewed distribution?** 
- We need to either apply a nonlinear map to the data make it symmetric or change the objective function. Which on is less painful?

**Change of Variable**: If the distribution over $X$ is $p_X(x)$, how the distribution over $y=f(x)$ is defined? 

$$
    y=f(x) \to x=f^{-1}(y) \\
    p_Y(y) = p_X(x)|\frac{\partial x}{\partial y}|_{x=f^-1(y)}
$$

**idea**: So, we need to find a function $f$ which make the $p_Y$ symmetric.

## Parametric Transformation 

- **Square Root Tranform**, $Y=\sqrt X$, suitable for right-skewed distribution. 
- **Log Transform**, $Y=\log (X+c)$, strongly right-skewed positive data. 
- **Reciprocal Transform**, $Y=\frac {1}{X + c}$, right-skewed data.
- **Box-Cox Transform**, positive continuous data.
$$
    Y(\lambda) = 
    \begin {cases}
            \frac {x^\lambda - 1}{\lambda}, \lambda \ne 0 \\
            \log X, \lambda = 0
    \end {cases}
$$
- **Yeo–Johnson transform**
$$
    Y(\lambda) = 
    \begin {cases}
        \frac {(x+1)^\lambda -1} {\lambda}, x \geq 0, \lambda \neq 0 \\
        \log (x+1), x \geq 0, \lambda = 0 \\
        -\frac {(1-x)^(2-\lambda) -1} {2-\lambda}, x \lt 0, \lambda \neq 2 \\
        \log (1-x), x \lt 0, \lambda=2

    \end {cases}
$$

## Histogram Equalization 

**Histogram equalization (HE)** - Enhance contrast or spread values uniformly across the available range. 

- For images $\to$ improves visibility of details.
- For general data $\to$ reduces skewness and spreads concentrated data regions

**Mathematical Principle**:
Let $X$ be the original variable (pixel intensity or numeric data), the CDF of $X$ is computed as 
$$
    F_X(x) = P(X \le x)
$$
As the CDF can be represented as an uniform distribution in interval $[0, 1]$, the map $Z=F_X(x)$ apply to values of $x$ can be regarrded as uniform distribution in the same interval, scaled to the actual range by $(L-1)Z$, where L is the scaling factor. 

# Categorical Inputs 
**Definition**: A categorical variable has a finite number of categories or labels for its values. For example, Gender is a categorical variable that can take "Male" and "Female" for its values.

## Types of Categorical Inputs
- **Nominal**: These are categorical variables whose values do not follow a natural order.
- **Ordinal**: These are categorical variables whose values follow a natural order. 

**Encoding schemas**:
- **Integer Encoding (Label Encoding)**: Assign a single integer to categories (red $\to$ 0, green $\to$ 1)
$$
    x_i \in X=\{x_1, x_2, ..., x_K\} \\
    \hat x = f(x_i) = i-1, ~~~ i=1, .., K
$$
- **One-hot Encoding**: A typical solution to treat the categorical features is to perform one-hot encoding, where if $x^{cat}_i $ can take $C$ values, then one-hot encoding transform data as:

$$
    \phi_{cat}: C \to \{0, 1\}^C \\
    \phi_{cat}(c) = e_j, ~~~~ j=index(c)
$$

- **Target Encoding**: Each category is replaced with the mean of the target variable for that category (regression task)
$$
    \hat x_i = E[y | x_i] = \frac {\Sigma_{j:x_j=c_i}y_j}{n_i}
$$

- **Weight of Evidence**: Used in binary classification, replacing a category with the log odds ratio of good vs bad outcomes. 

$$
    WoE(c_i) = \ln \frac {P(y=1|c_i)}{P(y=0|c_i)}
$$

# Design #1: Hashed Design Pattern

**Definition**:  
The **Hashed Design Pattern** is a method to handle **high-cardinality categorical inputs** by mapping them into a fixed-size numeric space using a hash function.  
- It addresses issues of memory, scalability, and cold-start in large datasets.  
- Especially useful when the vocabulary is too large or unknown values may appear in production.

---

## Issues with Traditional Encoding
1. **Incomplete Vocabulary (Cold Start)**  
   - One-hot encoding requires all categories to be known in advance.  
   - New or unseen categories during inference cause errors or require retraining.

2. **Model Size and High Cardinality**  
   - Large categorical features (millions of categories) lead to huge sparse matrices.  
   - Embedding layers may still require substantial memory if the cardinality is extremely high.

---

## Example Scenario
- **Task**: Predict whether a user will click on a product in an online retail platform.  
- **Problem**: High-cardinality features like `user_id` and `product_id` (millions of unique values).  
- **Solution**: Apply **hashed feature design pattern**:  
  1. Map categorical values into a **fixed-size bucket space** using a hash function.  
  2. Unseen categories are automatically assigned to a bucket without retraining.  
- **Benefits**:  
  - Reduces memory usage.  
  - Ensures scalability.  
  - Allows predictive modeling on large, real-world datasets.

---

## What is Hashing?
**Definition**: Process of converting data of arbitrary size into a **fixed-size output**.  

- Governed by a **hash function** with specific properties.  
- Two types of hashing:  
  1. **Cryptographic hashing**: Ensures security.  
  2. **Non-cryptographic / Fingerprint hashing**: Preserves similarity and is robust to minor input changes.  

> Fingerprint hashing is commonly used in hashed design patterns because it provides robust mappings for categorical features.

[More on Hash Functions](https://en.wikipedia.org/wiki/Hash_function)

---

## Steps for Creating Hashed Features
1. **Choose a hash function** $h$.  
2. **Determine the number of buckets** $D$.  
3. **Apply the hash function**: $h_i = h(x_i)$.  
4. **Map to bucket**:  
   $$
   \hat{x}_i = h_i \mod D
   $$  
   - Resulting values lie within $[0, D-1]$.  
   - Ensures a fixed-size numeric representation for each category.

> **Tip**: Always consider **training-serving skew** when hashing, so that the same mapping is used during inference.

---

## Key Considerations
- **Bucket Size \(D\)**:  
  - Critical parameter controlling **bucket collisions** (multiple categories mapping to the same bucket).  
  - Probability of collision:  
    $$
    p_{collision} = 1 - e^{-\frac{n(n-1)}{2D}}
    $$  
    where $n$ = number of unique categories, $D$ = bucket size.  
  - Larger bucket sizes reduce collisions but increase memory usage.

- **Other considerations**:  
  - Skewed distributions in the original categorical feature.  
  - Feature interactions may require additional handling (e.g., embedding crossed hashed features).  
  - Regularization may be needed to avoid overfitting on sparse or empty hashed buckets.

---

## When to Use
- Use hashed design pattern when:  
  - Vocabulary is extremely large.  
  - Cold-start problem is expected (new categories appear in production).  
  - Some bucket collisions are tolerable.  

- Avoid hashed design if:  
  - Vocabulary is small.  
  - No unseen categories are expected.  
  - High precision of categorical mapping is required.

---

## Summary
- Hashed design pattern **reduces high-cardinality categorical features** to a fixed-size numeric space.  
- Enables **scalable and memory-efficient** ML modeling.  
- Works best when combined with **embedding layers** or **regularization** to handle collisions and preserve information.  
- Key parameters: **bucket size**, **hash function**, and handling **feature interactions**.

# Design #2: Embedding Design Pattern

**Definition**:  
The **Embedding Design Pattern** is a method to automatically learn useful feature representations instead of relying on manual encoding.  
- It addresses the challenge of **high-cardinality categorical inputs** by mapping them into a lower-dimensional **dense vector space**.  
- Unlike fixed encodings or hashed features, embeddings are **learnable**, allowing the model to capture **similarity and relationships** between categories during training.  

---

## Embedding Layer
An **embedding layer** is a learnable mapping that converts discrete categorical inputs (e.g., integers representing categories) into **dense continuous vectors**.  

- Can be applied to **any categorical input**, not just text.  
- Maps discrete space → continuous space, making features **model-friendly**.  
- Preserves relationships among categories: similar categories can have **similar vector representations**.  

**Key Points**:  
- Semantic information is crucial in textual data but may be absent in generic categorical features.  
- Embeddings can reduce dimensionality while maintaining meaningful patterns.

---

## Text Embedding
Textual data is inherently **categorical with extremely high cardinality**, but also carries **semantic meaning**.  

**Tokenization**:  
- Converts text into smaller units (**tokens**), such as words, subwords, or characters.  
- Tokens are mapped to integer indices.  
- These indices are passed to an **embedding layer** to generate dense vectors capturing semantic relationships.  

**Example Applications**: NLP tasks like sentiment analysis, language modeling, and translation.

---

## Image Embedding
In computer vision, embeddings provide **compact feature representations** for images, allowing models to capture **complex input-output relationships**.  

- Replaces traditional hand-crafted features (edges, corners, textures) with **learned features**.  
- Enables similarity comparison, clustering, or feeding into downstream tasks like classification or object detection.

---

## Representation Learning
Both **text embeddings** and **image embeddings** are part of the broader field of **representation learning**:  
- The goal is to **encode raw inputs into meaningful, dense representations**.  
- Embedding layers are the **primary mechanism** for this in deep learning architectures.

---

## Key Advantages of Embedding Design Pattern
- Handles **high-cardinality categorical inputs** efficiently.  
- Captures **similarity and relationships** between categories.  
- Reduces dimensionality, making models more **efficient and generalizable**.  
- Can be combined with **feature crosses** or **hashed features** to handle extremely large categorical interactions.

---

## Representation Considerations 
- Not every task requires **complex embeddings** for unstructured data:  
  - Some tasks can work well with **simple representations** (e.g., TF-IDF for text, downsampled images, MFCCs for audio).  
  - Using embeddings adds **computational cost** and complexity.  
- The **choice of representation** depends on:  
  1. **Task complexity** – how subtle the patterns are.  
  2. **Expected model generalization** – whether the model needs to perform well on unseen data.  
  3. **Model capacity** – simpler models may not benefit from heavy embeddings.  
- **Rule of thumb**: start with simple representations; use embeddings only if the model **underfits** or **fails to generalize**.

---

## Summary
- Embeddings convert categorical or structured data into **dense vectors**.  
- They are **learnable**, allowing models to adapt representations during training.  
- Widely used in NLP, computer vision, and structured data tasks.  
- Complement other design patterns like **feature cross** and **hashed design pattern** to improve efficiency and accuracy in ML pipelines.


# Design #3: Feature Cross Design Pattern

**Definition**:  
The Feature Cross design pattern helps models learn relationships between inputs faster by explicitly creating new features that represent combinations of input values. This simplifies modeling and allows the model to capture interactions that might be difficult to learn automatically.

- **Purpose**:  
  - Capture interactions between features that a model might not easily learn on its own.  
  - While complex models like neural networks can learn interactions implicitly, feature crosses **make the modeling simpler and more explicit**, improving efficiency and potentially accuracy.

- **Categorical Inputs**:  
  - Feature crosses are **originally defined for categorical variables**, creating a new categorical feature from existing ones.  
  - The **cardinality** of the new feature = product of the cardinalities of the original features.  
  - High cardinality can lead to large feature spaces; to address this, it is recommended to use:  
    - **Embedding design pattern** – maps each category to a dense vector.  
    - **Hashed design pattern** – limits the cardinality by hashing categories into a fixed number of buckets.  
    - **Combination of embedding + hashing** – for large categorical interactions, enabling efficient training.

- **Numerical Inputs / Polynomial Interaction**:  
  - Some resources define feature cross for continuous values via **polynomial interaction**, e.g., $(x_1 \cdot x_2)$.  
  - Alternatively, **bucketizing continuous variables** before crossing allows the use of feature cross patterns in a categorical-style representation.  

- **Summary**:  
  1. Feature crosses **explicitly model interactions**.  
  2. High cardinality can occur, especially for categorical features.  
  3. Hashing and embeddings are standard solutions to manage dimensionality.  
  4. Continuous features require either **polynomial interactions** or **bucketization** to apply this pattern effectively.

# Design #4: Multi-modal Input
**Definition**: The **Multi-Modal Input Design Pattern** addresses the challenge of learning from datasets that contain **multiple types of data modalities** simultaneously, such as text, images, audio, or structured tabular data.  
- The goal is to create a unified model that can **effectively combine heterogeneous inputs** to improve prediction or representation learning.

---
## Motivation
- Real-world problems often involve **heterogeneous data sources**.  
  - Example: E-commerce platforms have user behavior logs (tabular), product images (visual), and reviews (text).  
- Many off-the-shelf models are **designed for a specific input type** only.  
  - Example: A standard image classification model such as ResNet-50 cannot handle text or structured data.  
- Single-modality models may **miss complementary information** contained in other modalities.  
- Multi-modal models allow:  
  - **Richer feature representation** by combining diverse inputs.  
  - **Improved accuracy and robustness** by leveraging multiple perspectives.  
  - **Flexibility** in handling datasets where different types of data coexist.  

**Example scenario**:  
- A traffic surveillance system: a camera captures footage at an intersection to detect violations.  
- The model must handle:  
  - **Image data** (camera footage).  
  - **Metadata** (time of day, day of week, weather).  
- To process non-numerical inputs like images or text, we use **embedding layers** to convert them into model-understandable vectors, then combine with structured tabular features.  

---
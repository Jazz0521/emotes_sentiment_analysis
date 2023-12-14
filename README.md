The repository contains the codes ,data and the lexica used for the sentiment analysis . 

Some observations and probable reasons of why does the performance of the average based classifier become worse after using the new emote dictionary
1) The key difference between the old and the new emote dictionary is that in the original emote dictionary ( containing 100 emotes ), the sentiment scores associated with each emote was a fraction between -1 to +1 , whereas in the new emote dictionary since the sentiments were only labelled as positive, negative or neutral - hence the sentiment scores were only either -1,0 or 1.

2) Average- Based Classifier - This classifier calculates sentiment by averaging the sentiment scores of individual tokens (words, emojis, emotes) in a sentence.
When we are using the original emote lexicon, the classifier can take advantage of the range of sentiment values (from -1 to 1) and can capture more nuanced sentiments, leading to potentially more accurate classification of sentiments in texts.
But when we switch to the second lexicon, which only has -1, 0, and 1 as sentiment values, the classifier loses the ability to capture subtle differences in sentiment. The averaging process becomes less informative because it operates on a less granular scale. This can lead to a decrease in recall, F1 score, and accuracy, as the classifier may not distinguish between texts with subtly different sentiments effectively. The finer gradations in the first lexicon allow for a more nuanced average sentiment score, which could be more accurate for certain types of text.

3) CNN-Based Classifier: Convolutional Neural Networks (CNNs) are capable of capturing contextual information and learning complex patterns in the data. This classifier does not rely directly on a sentiment lexicon for individual tokens. Instead, it learns from the overall structure and context of the input data. This makes it more adaptable and potentially more accurate when dealing with a larger and more varied set of emotes, as found in the second lexicon. Thus it is able to compensate for the lack of nuanced sentiment scores by analyzing the context in which emotes are used, thus improving the accuracy, recall, and F1 score.
The increase in recall, F1 score, and accuracy with the second lexicon also indicates that the CNN is effectively leveraging the larger number of emotes. The discrete sentiment values (-1, 0, 1) might be easier for the CNN to categorize, especially if it can extract meaningful features from the context in which the emotes are used. CNNs being more sophisticated and sensitive to contextual nuances than average-based models - hence they might be able to make better use of the additional emotes in the second lexicon, recognizing patterns that the simpler average-based model cannot.







The repository contains the codes ,data and the lexica used for the sentiment analysis . 

Probable reason of why does the performance of the average based classifier become worse after using the new dictionary ,
1) The key difference between the old and the new emote dictionary is that in the original emote dictionary ( containing 100 emotes ), the sentiment scores associated with each emote was a fraction between -1 to +1 , whereas in the new emote dictionary since the sentiments were only labelled as positive, negative or neutral - hence the sentiment scores were only -1,0 and 1

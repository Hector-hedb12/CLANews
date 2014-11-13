TweetsGetter
=======

Get tweets from the account specified in src/resources/input.txt and analize
each one:

a) Use a grammar-based tokenizer constructed with JFlex that:

  - Splits words at punctuation characters, removing punctuation. 
    However, a dot that's not followed by whitespace is considered 
    part of a token.
  - Splits words at hyphens, unless there's a number in the token, in 
    which case the whole token is interpreted as a product number and 
    is not split.
  - Recognizes email addresses and internet hostnames as one token. 

b) Normalizes tokens extracted with StandardTokenizer: remove 's and remove dots

c) Normalizes token text to lower case

d) Removes stop words from a token stream

e) Transforms the token stream as per the Porter stemming algorithm


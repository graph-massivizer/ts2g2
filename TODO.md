High level ideas:
 - **Time Series**: we have an abstraction for time series - we need to represent them
 - **Preprocessing**: when considering a time series, we may need to pre-process them. E.g., will we convert just a part of it to a graph? are we converting the time series time windows into graphs?
 - **Time series view**: any kind of time series preprocessing generates a view on the time series
 - **Strategies to generate Time Graphs**: we use strategies to create graphs that represent the information available in a time series view. 
 - **Time Graphs**: graphs representing a time series (e.g., NVG, etc.)
 - **To sequence strategies**: strategies to walk over the Time Graph and generate a sequence.
 - **To embedding visitor**: visitor to obtain a particular kind of embedding given a graph.


What needs to be done?
 - we need to decouple plotting from the model
 - we need to decouple representations that are specific for a kind of graph representation from the model. The internal states are currently messy, difficult to understand and maintain
 - we need to extract the components into separate files, to enhance readability
 - enhance documentation
 - update the examples and sample project.


 to run  tests
- python -m unittest discover tests

Raw Time Series → [Preprocessing] → Time Series View → [Graph Construction Strategy] → Time Graph → [Sequence/Embedding Extraction]


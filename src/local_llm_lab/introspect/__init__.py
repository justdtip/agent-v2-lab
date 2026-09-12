"""Training a model to report what has been added to its own residual stream.

Deliberately empty and lazy: importing this package must not drag in torch, a tokenizer or the
training path, because the data generator, the trainer and the evaluator each need a different
subset and the card cannot afford a module that imports all of them.
"""

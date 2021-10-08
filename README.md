# Check shape

TL;DR: This tool helps prevent bugs by checking shapes in-line, and making code more readable.

## Shape errors

A surprising amount of time in machine learning is spent on squashing shape-related bugs. There's two kinds of bugs:
* **Runtime shape errors**: These are annoying and can take up quite a bit of debugging time, but are not nearly as dangerous as broadcasting bugs.
* **Unindended broadcasting bugs**: These can be silent unexpected behaviours, which happen due to the broadcasting behaviour of the library in use.

I've found that one thing which goes a long way to prevent/solve this issues is putting shapes in the docstrings or with in-line comments, such as
```
def foo(bar1, bar2):
    """
    Does foo on bar1 and bar2.
    
    Arguments:
        bar1 : np.array, shape (B, D, 2)
        bar2 : np.array, shape (B, 5, D)
    """
    
    # Do an einsum, blip shape (B, 2, 5)
    blip = np.einsum('bdi, bjd -> bij', bar1, bar2)
```
Putting these shapes in is useful because it reduces the mental workload of remembering them, and improves readability. But **commented shapes are never enforced and could become stale**. When you read these docstrings/comments, you might **assume they are enforced**, when in fact they are not, causing unexpected broadcasting and weird errors. One way to enforce this is to use assertions, such as
```
def foo(bar1, bar2):
    """
    Does foo on bar1 and bar2.
    
    Arguments:
        bar1 : np.array, shape (B, D, 2)
        bar2 : np.array, shape (B, 5, D)
    """
    
    # Check that bar1 and bar2 are correctly shaped
    assert bar1.shape[0] == bar2.shape[0]
    assert bar1.shape[1] == bar2.shape[2]
    
    # Optionally, could also check the other dimensions
    assert bar1.shape[2] == 2 and bar2.shape[1] == 5
    
    # Do an einsum, blip shape (B, 2, 5)
    blip = np.einsum('bdi, bjd -> bij', bar1, bar2)
```
This does the job. It enforces the assumed shapes, and reduces the chance of a broadcasting error. But it can become very wordy and quite ugly when you start making more elaborate assertions, and can incur a mental load when reading code. Here's how to do the same thing with check_shape
```
def foo(bar1, bar2):
    """
    Does foo on bar1 and bar2.
    
    Arguments:
        bar1 : np.array, shape (B, D, 2)
        bar2 : np.array, shape (B, 5, D)
    """
    
    # Check shapes are compatible
    check_shape([bar1, bar2], [('B', 'D', 2), ('B', 5, 'D')])
    
    # Do an einsum, blip shape (B, 2, 5)
    blip = np.einsum('bdi, bjd -> bij', bar1, bar2)
```
Which (in my opinion) is more readable. The `check_shape` will raise a `ShapeError` whenever the arrays don't match the shapes given. Now you have a line of code which is both a shape-checking assertion, and also an inline comment -- which won't go stale!

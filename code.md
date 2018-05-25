---
layout: page
permalink: /code/
title: Coding Projects I Have Worked on
tags: [code]
modified: 24-5-2018
comments: false
---

"We build our computer (systems) the way we build our cities: over time, without a plan, on top of ruins." - Ellen Ullman
<br><br><br>
<p align="center"><img src="{{ site.url }}/images/CV/hosting.png" alt="Working Experience" width="90" height="90"></p>
* [**Dynamic HTML render and a Recursive Descendent Parser**](https://github.com/andreamad8/AP)<br>
In this work, we implement a library of Web Components similar to React.JS, which maintains its internal Virtual DOM representation, and optimized the render of the HTML page. Indeed, we implement functions which modifies only the parts
                      of the DOM element that has been changed since the previous rendering. Furthermore, we implemented a recursive descent parser that introduces the ability to express components in a style such as [JSX](https://facebook.github.io/jsx/#syntax),
                      reading from an input stream. The programming language used in this project is JavaScript.

<br><br>
<p align="center"><img src="{{ site.url }}/images/CV/book.png" alt="Working Experience" width="85" height="85"></p>

* [**Parallel implementation Jacobi Iterative Method**](https://github.com/andreamad8/jacobiMethod)<br>
The aim of this project is to parallelize the Jacobi Iterative Method. Implemented three versions of the code using C++,a sequential one and two parallel ones. The first parallel version has been implemented using [FastFlow](http://calvados.di.unipi.it/), and the second one using C++ pthread. Each code version has been evaluated using different matrix sizes, and measures (e.g., Completion Time, Scalability, Speed up, and Efficiency). All the conducted experiments have been run using a Xeon Phi coprocessor, which has 60 cores with 4 contexts each, clocked at 1GHz.


* [**PythonITA**](https://github.com/PisaCoderDojo/cpython-ita)<br>
This work is a group project of the Pisa CoderDojo and a batch of computer science students. Here, we fork the Cpython repository, and we extend the grammar of the Python languange in order to use Italian key words as native constructor.
                                                Furthermore, to fully integrate the new keys, we modify the language interpreter is some parts, such as the EBNF grammar (i.e. adding new production), the AST (abstract syntax tree), and the built-in functions. In addition, we modify the standard IDLE to have an editor that highlight and hints the newly added words. This project has been presented in the keynote of [Toscana DojoCon](https://coderdojo.com/news/2016/06/21/toscana-dojocon/),
                                                and it will be used for education's purpose as toy programming language.



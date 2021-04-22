#switch to global MathJax
find . -type f -name "*.html" -exec sed -i "s/<script type=\"text\/javascript\" src=\"\/usr\/share\/javascript\/mathjax\/MathJax\.js?config=TeX-AMS-MML_HTMLorMML\"><\/script>/<script id=\"MathJax-script\" src=\"https:\/\/cdn\.jsdelivr\.net\/npm\/mathjax\@3\/es5\/tex-chtml\.js\"><\/script>/g" {} +

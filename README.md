# Table of Contents

1.  [Article-recommender](#org91e20e7)
    1.  [Output](#org573123c)
    2.  [Documentation](#org629cc6e)
    3.  [How to package](#org4435ec6)


<a id="org91e20e7"></a>

# Article-recommender

article normalized scores generator


<a id="org573123c"></a>

## Output

The output looks something like this:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">wikidata_id</th>
<th scope="col" class="org-right">normalized_rank</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Q125576</td>
<td class="org-right">0.930232</td>
</tr>


<tr>
<td class="org-left">Q127418</td>
<td class="org-right">0.928457</td>
</tr>


<tr>
<td class="org-left">Q125576</td>
<td class="org-right">0.927625</td>
</tr>


<tr>
<td class="org-left">Q226697</td>
<td class="org-right">0.919053</td>
</tr>


<tr>
<td class="org-left">&#x2026;</td>
<td class="org-right">&#xa0;</td>
</tr>
</tbody>
</table>


<a id="org629cc6e"></a>

## Documentation

Code documentation adheres to the Google Style Python
Docstrings<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>.

For other documentation see the `doc` folder.


<a id="org4435ec6"></a>

## How to package

-   From the root directory run:
    -   python3 setup.py sdist bdist_wheel
-   Generated files will be in ./dist/


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>

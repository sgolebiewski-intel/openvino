import os.path
import requests
import json
import re

from sphinx.directives.code import LiteralInclude, LiteralIncludeReader, container_wrapper
from sphinx.util import logging
from docutils.parsers.rst import Directive, directives
from typing import List, Tuple
from docutils.nodes import Node
from docutils import nodes
from sphinx.util import parselinenos
from pathlib import Path
from bs4 import BeautifulSoup as bS

logger = logging.getLogger(__name__)


class DoxygenSnippet(LiteralInclude):

    option_spec = dict({'fragment': directives.unchanged_required}, **LiteralInclude.option_spec)

    def run(self) -> List[Node]:
        if 'fragment' in self.options:
            self.options['start-after'] = self.options['fragment']
            self.options['end-before'] = self.options['fragment']
        document = self.state.document
        if not document.settings.file_insertion_enabled:
            return [document.reporter.warning('File insertion disabled',
                                              line=self.lineno)]
        # convert options['diff'] to absolute path
        if 'diff' in self.options:
            _, path = self.env.relfn2path(self.options['diff'])
            self.options['diff'] = path

        try:
            location = self.state_machine.get_source_and_line(self.lineno)
            doxygen_snippet_root = self.config.html_context.get('doxygen_snippet_root')

            if doxygen_snippet_root and os.path.exists(doxygen_snippet_root):
                rel_filename = self.arguments[0]
                filename = os.path.join(doxygen_snippet_root, rel_filename)
            else:
                rel_filename, filename = self.env.relfn2path(self.arguments[0])
            self.env.note_dependency(rel_filename)

            reader = LiteralIncludeReader(filename, self.options, self.config)
            text, lines = reader.read(location=location)

            retnode = nodes.literal_block(text, text, source=filename)  # type: Element
            retnode['force'] = 'force' in self.options
            self.set_source_info(retnode)
            if self.options.get('diff'):  # if diff is set, set udiff
                retnode['language'] = 'udiff'
            elif 'language' in self.options:
                retnode['language'] = self.options['language']
            if ('linenos' in self.options or 'lineno-start' in self.options or
                    'lineno-match' in self.options):
                retnode['linenos'] = True
            retnode['classes'] += self.options.get('class', [])
            extra_args = retnode['highlight_args'] = {}
            if 'emphasize-lines' in self.options:
                hl_lines = parselinenos(self.options['emphasize-lines'], lines)
                if any(i >= lines for i in hl_lines):
                    logger.warning(__('line number spec is out of range(1-%d): %r') %
                                   (lines, self.options['emphasize-lines']),
                                   location=location)
                extra_args['hl_lines'] = [x + 1 for x in hl_lines if x < lines]
            extra_args['linenostart'] = reader.lineno_start

            if 'caption' in self.options:
                caption = self.options['caption'] or self.arguments[0]
                retnode = container_wrapper(self, retnode, caption)

            # retnode will be note_implicit_target that is linked from caption and numref.
            # when options['name'] is provided, it should be primary ID.
            self.add_name(retnode)

            return [retnode]
        except Exception as exc:
            return [document.reporter.warning(exc, line=self.lineno)]


def visit_scrollbox(self, node):
    attrs = {}
    attrs["style"] = (
        (("height:" + "".join(c for c in str(node["height"]) if c.isdigit()) + "px!important; " ) if "height" in node is not None else "")
        + (("width:" + "".join(c for c in str(node["width"]) if c.isdigit()) ) if "width" in node is not None else "")
        + (("px; " if node["width"].find("px") != -1 else "%;") if "width" in node is not None else "")
        + ( ("border-left:solid "+"".join(c for c in str(node["delimiter"]) if c.isdigit())+ "px " + (("".join(str(node["delimiter-color"]))) if "delimiter-color" in node is not None else "#dee2e6") +"; ") if "delimiter" in node is not None else "")
    )
    attrs["class"] = "scrollbox"
    self.body.append(self.starttag(node, "div", **attrs))


def depart_scrollbox(self, node):
    self.body.append("</div>\n")


class Nodescrollbox(nodes.container):
    def create_scrollbox_component(
        rawtext: str = "",
        **attributes,
    ) -> nodes.container:
        node = nodes.container(rawtext, is_div=True, **attributes)
        return node


class Scrollbox(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        'name': directives.unchanged,
        'width': directives.length_or_percentage_or_unitless,
        'height': directives.length_or_percentage_or_unitless,
        'style': directives.unchanged,
        'delimiter': directives.length_or_percentage_or_unitless,
        'delimiter-color': directives.unchanged,
    }

    has_content = True

    def run(self):
        classes = ['scrollbox','']
        node = Nodescrollbox("div", rawtext="\n".join(self.content), classes=classes)
        if 'height' in self.options:
            node['height'] = self.options['height']
        if 'width' in self.options:
            node['width'] = self.options['width']
        if 'delimiter' in self.options:
            node['delimiter'] = self.options['delimiter']
        if 'delimiter-color' in self.options:
            node['delimiter-color'] = self.options['delimiter-color']
        self.add_name(node)
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def fetch_binder_list(file) -> list:
    with open(file, 'r+', encoding='cp437') as file:
        list_of_buttons = file.read().splitlines()
    return list_of_buttons


def fetch_colab_list(file) -> list:
        with open(file, 'r+', encoding='cp437') as file:
            list_of_cbuttons = file.read().splitlines()
        return list_of_cbuttons

def visit_showcase(self, node):
    attrs = {}
    notebooks_repo = "https://github.com/openvinotoolkit/openvino_notebooks/blob/main/"
    notebooks_binder = "https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F"
    notebooks_colab = "https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/"
    git_badge = "<img class='showcase-badge' src='https://badgen.net/badge/icon/github?icon=github&amp;label' alt='Github'>"
    binder_badge = "<img class='showcase-badge' src='https://mybinder.org/badge_logo.svg' alt='Binder'>"
    colab_badge = "<img class='showcase-badge' src='https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667' alt='Colab'>"
    parent_repo_name = "openvinotoolkit"
    notebooks_repo_link = "openvino_notebooks"
    github_api_link = "https://api.github.com/repos/{}/{}/git/trees/main?recursive=1".format(parent_repo_name,
                                                                                             notebooks_repo_link)
    #doc_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
    binder_path = Path('../../../docs/notebooks/notebooks_with_binder_buttons.txt').resolve(strict=True)
    colab_path = Path('../../../docs/notebooks/notebooks_with_colab_buttons.txt').resolve(strict=True)
    #notebooks_dir = "/notebooks/"
    #binder_list_file = "notebooks_with_binder_buttons.txt"
    #colab_list_file = "notebooks_with_colab_buttons.txt"
    # jsonfile = doc_dir + notebooks_dir + "main.json"
    #binder_path = doc_dir + notebooks_dir + binder_list_file
    #colab_path = doc_dir + notebooks_dir + colab_list_file
    binder_buttons_list = fetch_binder_list(binder_path)
    colab_buttons_list = fetch_colab_list(colab_path)
    result = requests.get(github_api_link)
    # result = open(jsonfile, 'r').read()
    parse_tree = bS(result.text, 'html.parser')
    # parse_tree = bS(result, 'html.parser')
    paths_list = json.loads(parse_tree.text)
    list_all_paths = [p.get('path') for p in paths_list['tree'] if p.get('path')]
    ipynb_ext = re.compile(".*\\.ipynb")
    ipynb_list = list(filter(ipynb_ext.match, list_all_paths))
    notebook_with_ext = node["title"] + ".ipynb"
    matched_notebook = [match for match in ipynb_list if notebook_with_ext in match]

    if "height" or "width" in node:
        attrs["style"] = (
            (("height:" + "".join(c for c in str(node["height"]) if c.isdigit()) + "px!important; " ) if "height" in node is not None else "")
            + (("width:" + "".join(c for c in str(node["width"]) if c.isdigit()) ) if "width" in node is not None else "")
            + (("px; " if node["width"].find("px") != -1 else "%;") if "width" in node is not None else "")
        )
    self.body.append("<div class='showcase-wrap'>")
    self.body.append(self.starttag(node, "div", **attrs))
    self.body.append(("<div class='showcase-img-placeholder'><img " + (" class='" + (node["img-class"] + " showcase-img' ") if 'img-class' in node is not None else " class='showcase-img'") + "src='" + node["img"] + "' alt='"+os.path.basename(node["img"])+"' /></div>") if "img" in node is not None else "")
    self.body.append(("<div class='showcase-content'><div class='showcase-content-container'><h2 class='showcase-title'>" + node["title"] + "</h2>") if 'title' in node is not None else "")

    if matched_notebook is not None:
        for n in matched_notebook:
            self.body.append(("<a href='" + notebooks_repo + n + "' target='_blank'>" + git_badge + "</a>") if 'title' in node is not None else "")
            if node["title"] in binder_buttons_list:
                self.body.append(("<a href='" + notebooks_binder + n + "' target='_blank'>" + binder_badge + "</a>"
                                  ) if 'title' in node is not None else "")
            if node["title"] in colab_buttons_list:
                self.body.append(("<a href='" + notebooks_colab + n + "' target='_blank'>" + colab_badge + "</a>"
                                  ) if 'title' in node is not None else "")


def depart_showcase(self, node):
    notebook_file = ("notebooks/" + node["title"] + "-with-output.html") if 'title' in node is not None else ""
    link_title = (node["title"]) if 'title' in node is not None else "OpenVINO Interactive Tutorial"
    self.body.append("</div><button class='showcase-button' type='button' title='" + link_title +
                     "' onclick=\"location.href='" + notebook_file + "'\">Read more</a></div></div></div>\n")


class Nodeshowcase(nodes.container):
    def create_showcase_component(
        rawtext: str = "",
        **attributes,
    ) -> nodes.container:
        node = nodes.container(rawtext, is_div=True, **attributes)
        return node


class Showcase(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        'class': directives.class_option,
        'name': directives.unchanged,
        'width': directives.length_or_percentage_or_unitless,
        'height': directives.length_or_percentage_or_unitless,
        'style': directives.unchanged,
        'img': directives.unchanged,
        'img-class': directives.unchanged,
        'title': directives.unchanged,
        'git': directives.unchanged,
    }

    has_content = True

    def run(self):

        classes = ['showcase']
        node = Nodeshowcase("div", rawtext="\n".join(self.content), classes=classes)
        if 'height' in self.options:
            node['height'] = self.options['height']
        if 'width' in self.options:
            node['width'] = self.options['width']
        if 'img' in self.options:
            node['img'] = self.options['img']
        if 'img-class' in self.options:
            node['img-class'] = self.options['img-class']
        if 'title' in self.options:
            node['title'] = self.options['title']
        if 'git' in self.options:
            node['git'] = self.options['git']
        node['classes'] += self.options.get('class', [])
        self.add_name(node)
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

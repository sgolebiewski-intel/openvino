import os
import json
import sys
from json import JSONDecodeError
from sphinx.errors import ExtensionError
import jinja2
from docutils.parsers import rst
from docutils.parsers.rst import roles
from docutils import nodes
from pathlib import Path
from sphinx.util import logging
from .directives.code import DoxygenSnippet, Scrollbox, Nodescrollbox, visit_scrollbox, depart_scrollbox
import re
import subprocess

SPHINX_LOGGER = logging.getLogger(__name__)


def setup_edit_url(app, pagename, templatename, context, doctree):
    """Add a function that jinja can access for returning the edit URL of a page."""

    def has_github_page():
        doxygen_mapping_file = app.config.doxygen_mapping_file
        if os.path.basename(pagename) in doxygen_mapping_file:
            return True
        return False

    def get_edit_url():
        """Return a URL for an "Edit on GitHub" link."""
        doc_context = dict()
        doc_context.update(**context)

        # ensure custom URL is checked first, if given
        url_template = doc_context.get("edit_page_url_template")

        if url_template is not None:
            if "file_name" not in url_template:
                raise ExtensionError(
                    "Missing required value for `use_edit_page_button`. "
                    "Ensure `file_name` appears in `edit_page_url_template`: "
                    f"{url_template}"
                )
            return jinja2.Template(url_template).render(**doc_context)

        url_template = '{{ github_url }}/{{ github_user }}/{{ github_repo }}' \
                       '/edit/{{ github_version }}/{{ file_name }}'

        doxygen_mapping_file = app.config.doxygen_mapping_file
        rst_name = os.path.basename(pagename)
        file_name = doxygen_mapping_file[rst_name]
        parent_folder = Path(os.path.dirname(file_name)).parts[0]
        file_name = Path(*Path(file_name).parts[1:]).as_posix()

        doc_context.update(file_name=file_name)
        try:
            repositories = app.config.repositories
        except AttributeError:
            raise ExtensionError("Missing required value for `use_edit_page_button`. "
                                 "Ensure `repositories` is set in conf.py.")

        required = ['github_user', 'github_repo', 'github_version', 'host_url']
        for repo, config in repositories.items():
            for key, val in config.items():
                if key not in required or not val:
                    raise ExtensionError(f'Missing required value for `{repo}` entry in `repositories`'
                                         f'Ensure {required} all set.')
            if parent_folder == repo:
                doc_context.update(github_user=config['github_user'])
                doc_context.update(github_repo=config['github_repo'])
                doc_context.update(github_version=config['github_version'])
                doc_context.update(github_url=config['host_url'])
                return jinja2.Template(url_template).render(**doc_context)

    context["get_edit_url"] = get_edit_url
    context['has_github_page'] = has_github_page()

    # Ensure that the max TOC level is an integer
    context["theme_show_toc_level"] = int(context.get("theme_show_toc_level", 1))


def get_theme_path():
    theme_path = os.path.abspath(os.path.dirname(__file__))
    return theme_path


def read_doxygen_configs(app, env, docnames):
    if app.config.html_context.get('doxygen_mapping_file'):
        try:
            with open(app.config.html_context.get('doxygen_mapping_file'), 'r', encoding='utf-8') as f:
                app.config.html_context['doxygen_mapping_file'] = json.load(f)
        except (JSONDecodeError, FileNotFoundError):
            app.config.html_context['doxygen_mapping_file'] = dict()

def get_branch_name():
    branch_name = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD']).strip().decode()
    if not branch_name:
        raise Exception("This is neither a valid branch name nor any repository.", branch_name)
    return branch_name

def link_to_repo(repo_file_path):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        title_only = re.compile("<.*?>")
        title = title_only.sub('', text)
        path = text[text.find("<")+1:text.find(">")]
        url = repo_file_path % (path,)
        node = nodes.reference(rawtext, title, refuri=url, **options)
        return [node], []
    return role

ov_repo_link = 'https://github.com/openvinotoolkit/openvino'
ovms_repo_link = 'https://github.com/openvinotoolkit/model_server'
current_branch = get_branch_name()
roles.register_canonical_role('ovlink', link_to_repo('{}/blob/{}/%s'.format(ov_repo_link, current_branch)))
roles.register_canonical_role('ovmslink', link_to_repo('{}/blob/{}/%s'.format(ovms_repo_link, current_branch)))

def setup(app):
    theme_path = get_theme_path()
    templates_path = os.path.join(theme_path, 'templates')
    static_path = os.path.join(theme_path, 'static')
    app.config.templates_path.append(templates_path)
    app.config.html_static_path.append(static_path)
    app.connect("html-page-context", setup_edit_url, priority=sys.maxsize)
    app.connect('env-before-read-docs', read_doxygen_configs)
    app.add_html_theme('openvino_sphinx_theme', theme_path)
    rst.directives.register_directive('doxygensnippet', DoxygenSnippet)
    rst.directives.register_directive('scrollbox', Scrollbox)
    app.add_node(
        Nodescrollbox,
        html=(visit_scrollbox, depart_scrollbox),
        latex=(visit_scrollbox, depart_scrollbox)
    )
    return {'parallel_read_safe': True, 'parallel_write_safe': True}

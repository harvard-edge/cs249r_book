"""
Sphinx extension to resolve icon_links URLs using pathto() for relative paths.

This extension hooks into the html-page-context event to modify icon_links URLs
that start with '_static' to use proper relative paths from the current page.

This solves the problem of navbar icon links (like PDF downloads) breaking when
accessed from subdirectories like modules/, tiers/, or tito/.

Usage in _config.yml:
    sphinx:
      local_extensions:
        icon_link_resolver: ./_ext
"""

from sphinx.application import Sphinx


def resolve_icon_links(app, pagename, templatename, context, doctree):
    """
    Hook into html-page-context to resolve icon_links URLs.

    For URLs starting with '_static', convert them to proper relative paths
    using the pathto() function available in the context.
    """
    theme_icon_links = context.get('theme_icon_links', [])

    if not theme_icon_links:
        return

    pathto = context.get('pathto')
    if not pathto:
        return

    # Create a new list with resolved URLs
    resolved_links = []
    for link in theme_icon_links:
        new_link = dict(link)  # Copy the link dict
        url = link.get('url', '')

        # If URL starts with _static, resolve it using pathto
        if url.startswith('_static'):
            new_link['url'] = pathto(url, 1)

        resolved_links.append(new_link)

    # Update the context with resolved links
    context['theme_icon_links'] = resolved_links


def setup(app: Sphinx):
    """Register the extension with Sphinx."""
    app.connect('html-page-context', resolve_icon_links)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

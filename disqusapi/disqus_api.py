"""
disqus-python
~~~~~~~~~~~~~

disqus = DisqusAPI(api_secret=secret_key)
disqus.get('trends.listThreads')

"""
try:
    __version__ = __import__('pkg_resources') \
        .get_distribution('disqusapi').version
except Exception:  # pragma: no cover
    __version__ = 'unknown'

import re
import zlib
import os.path
import warnings
import socket

try:
    import simplejson as json
except ImportError:
    import json

from disqusapi.paginator import Paginator
from disqusapi import compat
from disqusapi.compat import http_client as httplib
from disqusapi.compat import urllib_parse as urllib
from disqusapi.utils import build_interfaces_by_method

__all__ = ['DisqusAPI', 'Paginator']

with open(os.path.join(os.path.dirname(__file__), 'interfaces.json')) as fp:
    INTERFACES = json.load(fp)

HOST = 'disqus.com'

CHARSET_RE = re.compile(r'charset=(\S+)')
DEFAULT_ENCODING = 'utf-8'


class InterfaceNotDefined(NotImplementedError):
    pass


class InvalidHTTPMethod(TypeError):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "expected 'GET' or 'POST', got: %r" % self.message


class FormattingError(ValueError):
    pass


class APIError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return '%s: %s' % (self.code, self.message)


class InvalidAccessToken(APIError):
    pass

ERROR_MAP = {
    18: InvalidAccessToken,
}


class Result(object):
    def __init__(self, response, cursor=None):
        self.response = response
        self.cursor = cursor or {}

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, repr(self.response))

    def __iter__(self):
        for r in self.response:
            yield r

    def __len__(self):
        return len(self.response)

    def __getslice__(self, i, j):
        return list.__getslice__(self.response, i, j)

    def __getitem__(self, key):
        return list.__getitem__(self.response, key)

    def __contains__(self, key):
        return list.__contains__(self.response, key)


class DisqusAPI():
    formats = {
        'json': (json.loads, ValueError),
    }

    def __init__(self, secret_key=None, public_key=None, format='json', version='3.0',
                 timeout=None, interfaces=INTERFACES, **kwargs):
        self.secret_key = secret_key
        self.public_key = public_key
        self.format = format
        self.version = version
        self.timeout = timeout or socket.getdefaulttimeout()
        self.interfaces = interfaces
        self.interfaces_by_method = build_interfaces_by_method(self.interfaces)

    def __getattr__(self, attr):
        if attr in getattr(self, '__dict__'):
            return getattr(self, attr)
        interface = {}
        try:
            interface = self.interfaces[attr]
        except KeyError:
            try:
                interface = self.interfaces_by_method[attr]
            except KeyError:
                pass

    def request(self, endpoint=None, **kwargs):
        if endpoint is not None:
            domain, req = endpoint.split('.')
            resource = self.interfaces.get(domain, {}).get(req, {})
            endpoint = endpoint.replace('.', '/')

        for k in resource.get('required', []):
            if k not in (x.split(':')[0] for x in compat.iterkeys(kwargs)):
                raise ValueError('Missing required argument: %s' % k)

        method = kwargs.pop('method', resource.get('method'))

        if not method:
            raise InterfaceNotDefined(
                'Interface is not defined, you must pass ``method`` (HTTP Method).')

        method = method.upper()
        if method not in ('GET', 'POST'):
            raise InvalidHTTPMethod(method)


        version = kwargs.pop('version', self.version)
        format = kwargs.pop('format', self.format)
        formatter, formatter_error = self.formats[format]

        path = '/api/%s/%s.%s' % (version, endpoint, format)

        if 'api_secret' not in kwargs and self.secret_key:
            kwargs['api_secret'] = self.secret_key
        if 'api_public' not in kwargs and self.public_key:
            kwargs['api_key'] = self.public_key

        # We need to ensure this is a list so that
        # multiple values for a key work
        params = []
        for k, v in compat.iteritems(kwargs):
            if isinstance(v, (list, tuple)):
                for val in v:
                    params.append((k, val))
            else:
                params.append((k, v))

        headers = {
            'User-Agent': 'disqus-python/%s' % __version__,
            'Accept-Encoding': 'gzip',
        }

        if method == 'GET':
            path = '%s?%s' % (path, urllib.urlencode(params))
            data = ''
        else:
            data = urllib.urlencode(params)
            headers['Content-Type'] = 'application/x-www-form-urlencoded'

        conn = httplib.HTTPSConnection(HOST, timeout=self.timeout)
        conn.request(method, path, data, headers)
        response = conn.getresponse()

        try:
            body = response.read()
        finally:
            # Close connection
            conn.close()

        if response.getheader('Content-Encoding') == 'gzip':
            # See: http://stackoverflow.com/a/2424549
            body = zlib.decompress(body, 16 + zlib.MAX_WBITS)

        # Determine the encoding of the response and respect
        # the Content-Type header, but default back to utf-8
        content_type = response.getheader('Content-Type')
        if content_type is None:
            encoding = DEFAULT_ENCODING
        else:
            try:
                encoding = CHARSET_RE.search(content_type).group(1)
            except AttributeError:
                encoding = DEFAULT_ENCODING

        body = body.decode(encoding)

        try:
            # Coerce response to Python
            data = formatter(body)
        except formatter_error:
            raise FormattingError(body)

        if response.status != 200:
            raise ERROR_MAP.get(data['code'], APIError)(data['code'], data['response'])

        if isinstance(data['response'], list):
            return Result(data['response'], data.get('cursor'))
        return data['response']

    def update_interface(self, new_interface):
        self.interfaces.update(new_interface)
        self.interfaces_by_method = build_interfaces_by_method(self.interfaces)

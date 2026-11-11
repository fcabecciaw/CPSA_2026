"""Client-server protocol v1: message builders (interface specification IF-1xx..4xx).

Pure code: builds ``data`` payloads from domain and vision types. The envelope is
added by the server adapter per connection.
"""

PROTOCOL_VERSION = 1

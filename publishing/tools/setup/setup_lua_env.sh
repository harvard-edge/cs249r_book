#!/bin/bash

# Setup script for Lua environment to work with cross-reference filter
# This sets up the paths so that the dkjson module can be found by Quarto

export LUA_PATH='/usr/local/Cellar/luarocks/3.12.2/share/lua/5.4/?.lua;/usr/local/share/lua/5.4/?.lua;/usr/local/share/lua/5.4/?/init.lua;/usr/local/lib/lua/5.4/?.lua;/usr/local/lib/lua/5.4/?/init.lua;./?.lua;./?/init.lua;/Users/VJ/.luarocks/share/lua/5.4/?.lua;/Users/VJ/.luarocks/share/lua/5.4/?/init.lua'

export LUA_CPATH='/usr/local/lib/lua/5.4/?.so;/usr/local/lib/lua/5.4/loadall.so;./?.so;/Users/VJ/.luarocks/lib/lua/5.4/?.so'

echo "✅ Lua environment set up for cross-reference filter"
echo "🔗 You can now run: quarto render [file] --to [format]"
echo "📄 Cross-references will be automatically injected during rendering"

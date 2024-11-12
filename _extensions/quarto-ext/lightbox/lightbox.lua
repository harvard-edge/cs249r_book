-- whether we're automatically lightboxing
local auto = false

-- whether we need lightbox dependencies added
local needsLightbox = false

-- a counter used to ensure each image is in its own gallery
local imgCount = 0

-- attributes to forward from the image to the newly created link
local kDescription = "description"
local kForwardedAttr = {
  "title", kDescription, "desc-position", 
  "type", "effect", "zoomable", "draggable"
}

local kLightboxClass = "lightbox"
local kNoLightboxClass = "nolightbox"
local kGalleryPrefix = "quarto-lightbox-gallery-"

-- A list of images already within links that we can use to filter
local imagesWithinLinks = pandoc.List({})

local function readAttrValue(el, attrName) 
  if attrName == kDescription then
    local doc = pandoc.read(el.attr.attributes[attrName])
    local attrInlines = doc.blocks[1].content
    return pandoc.write(pandoc.Pandoc(attrInlines), "html")
  else 
    return el[attrName]
  end

end

return {
  {
    Meta = function(meta) 

      -- If the mode is auto, we need go ahead and 
      -- run if there are any images (ideally we would)
      -- filter to images in the body, but that can be
      -- left for future me to deal with 
      -- supports:
      -- lightbox: auto
      -- or
      -- lightbox:
      --   match: auto
      local lbMeta = meta.lightbox
      if lbMeta ~= nil and type(lbMeta) == 'table' then
        if lbMeta[1] ~= nil then
          if lbMeta[1]['text'] == "auto" then
            auto = true
          end
        elseif lbMeta.match ~= nil and pandoc.utils.stringify(lbMeta.match) == 'auto' then
          auto = true
        elseif lbMeta == true then
          auto = true      
        end
      end
    end, 
    -- Find images that are already within links
    -- we'll use this to filter out these images if
    -- the most is auto
    Link = function(linkEl)
      pandoc.walk_inline(linkEl, {
        Image = function(imageEl) 
          imagesWithinLinks[#imagesWithinLinks + 1] = imageEl
        end
      })
    end
  },{
    Div = function(div)
      if div.classes:includes("cell") and div.attributes["lightbox"] ~= nil then
        meta = quarto.json.decode(div.attributes["lightbox"])
        local imgCount=0
        div = div:walk({
          Image = function(imgEl)
            imgCount = imgCount + 1
            if (type(meta) == "table" and meta[kNoLightboxClass] == true) or meta == false then
              imgEl.classes:insert(kNoLightboxClass)
            else
              if not auto and ((type(meta) == "table" and not meta[kNoLightboxClass]) or meta == true) then
                imgEl.classes:insert(kLightboxClass)
              end
              if (type(meta) == "table") then
                if meta.group then
                  imgEl.attr.attributes.group = meta.group or imgEl.attr.attributes.group
                end
                for _, v in next, kForwardedAttr do
                  if type(meta[v]) == "table" and #meta[v] > 1 then 
                    -- if list attributes it should be one per plot
                    if imgCount > #meta[v] then
                      quarto.log.warning("More plots than '" .. v .. "' passed in YAML chunk options.")
                    else
                      attrLb = meta[v][imgCount]
                    end
                  else 
                    -- Otherwise reuse the single attributes
                    attrLb = meta[v]
                  end
                  imgEl.attr.attributes[v] = attrLb or imgEl.attr.attributes[v]
                end
              end
            end
            return imgEl
          end
        })
        div.attributes["lightbox"] = nil
      end
      return div
    end
  },
  {
  Image = function(imgEl)
    if quarto.doc.is_format("html:js") then
      local isAlreadyLinked = imagesWithinLinks:includes(imgEl)
      if (not isAlreadyLinked and auto and not imgEl.classes:includes(kNoLightboxClass)) 
          or imgEl.classes:includes('lightbox') then
        -- note that we need to include the dependency for lightbox
        needsLightbox = true
        imgCount = imgCount + 1

        -- remove the class from the image
        imgEl.attr.classes = imgEl.attr.classes:filter(function(clz) 
          return clz ~= kLightboxClass
        end)
        
        -- attributes for the link
        local linkAttributes = {}

        -- mark this image as a lightbox target
        linkAttributes.class = kLightboxClass

        -- get the alt text from image and use that as title
        local title = nil
        if imgEl.caption ~= nil and #imgEl.caption > 0 then
          linkAttributes.title = pandoc.utils.stringify(imgEl.caption)
        elseif imgEl.attributes['fig-alt'] ~= nil and #imgEl.attributes['fig-alt'] > 0 then
          linkAttributes.title = pandoc.utils.stringify(imgEl.attributes['fig-alt'])
        end

        -- move a group attribute to the link, if present
        if imgEl.attr.attributes.group ~= nil then
          linkAttributes.gallery = imgEl.attr.attributes.group
          imgEl.attr.attributes.group = nil
        else 
          linkAttributes.gallery = kGalleryPrefix .. imgCount
        end

        -- forward any other known attributes
        for i, v in ipairs(kForwardedAttr) do
          if imgEl.attr.attributes[v] ~= nil then
            -- forward the attribute
            linkAttributes[v] = readAttrValue(imgEl, v)
          
            -- clear the attribute
            imgEl.attr.attributes[v] = nil
          end

          -- clear the title
          if (imgEl.title == 'fig:') then
            imgEl.title = ""
          end

        end

        -- wrap decorated images in a link with appropriate attrs
        local link = pandoc.Link({imgEl}, imgEl.src, nil, linkAttributes)
        return link
      end
    end 
  end,
  Meta = function(meta)
    -- If we discovered lightbox-able images
    -- we need to include the dependencies
    if needsLightbox then
      -- add the dependency
      quarto.doc.add_html_dependency({
        name = 'glightbox',
        scripts = {'resources/js/glightbox.min.js'},
        stylesheets = {'resources/css/glightbox.min.css', 'lightbox.css'}
      })

      -- read lightbox options
      local lbMeta = meta.lightbox
      local lbOptions = {}
      local readEffect = function(el) 
        local val = pandoc.utils.stringify(el)
        if val == "fade" or val == "zoom" or val == "none" then
          return val
        else
          error("Invalid effect " + val)
        end
      end

      -- permitted options include:
      -- lightbox:
      --   effect: zoom | fade | none
      --   desc-position: top | bottom | left |right
      --   loop: true | false
      --   class: <class-name>
      local effect = "zoom"
      local descPosition = "bottom" 
      local loop = true
      local skin = nil
      
      -- The selector controls which elements are targeted.
      -- currently, it always targets .lightbox elements
      -- and there is no way for the user to change this
      local selector = "." .. kLightboxClass

      if lbMeta ~= nil and type(lbMeta) == 'table' then
        if lbMeta.effect ~= nil then
          effect = readEffect(lbMeta.effect)
        end

        if lbMeta['desc-position'] ~= nil then
          descPosition = pandoc.utils.stringify(lbMeta['desc-position'])
        end  

        if lbMeta['css-class'] ~= nil then
          skin = pandoc.utils.stringify(lbMeta['css-class'])
        end
        
        if lbMeta.loop ~= nil then
          loop = lbMeta.loop
        end
      end

      -- Generate the options to configure lightbox
      local options = {
        selector = selector,
        closeEffect = effect,
        openEffect = effect, 
        descPosition = descPosition,
        loop = loop,
      }
      if skin ~= nil then
        options.skin = skin
      end
      local optionsJson = quarto.json.encode(options)

      -- generate the initialization script with the correct options
      local scriptTag = "<script>var lightboxQuarto = GLightbox(" .. optionsJson .. ");</script>"

      -- inject the rendering code
      quarto.doc.include_text("after-body", scriptTag)

    end
  end
}}

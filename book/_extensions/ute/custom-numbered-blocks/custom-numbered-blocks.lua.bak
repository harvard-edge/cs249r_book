--[[
MIT License

Copyright (c) 2023 Ute Hahn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]--
-- pre-pre-release
-- 
-- partial rewrite, complete later

-- nice rename function learned from shafayetShafee :-)
local str = pandoc.utils.stringify
local pout = quarto.log.output

-- important quasi global variables

local ishtml = quarto.doc.is_format("html")
local ispdf = quarto.doc.is_format("pdf")
local fmt=""
if ishtml then fmt="html" elseif ispdf then fmt = "pdf" end

-- TODO encapsulate stylez into a doc thing or so
-- maybe later allow various versions concurrently. 
-- this could probably be problematic because of option clashes in rendered header?

local stylename="foldbox"

local stylez = require("style/"..stylename)


--- TODO: better encapsulation (luxury :-P)

fbx={ -- global table, holds information for processing fboxes
   xreffile = "._xref.json" -- default name, set to lastfile in initial meta analysis
}


-- utility functions ---
local function DeInline(tbl)  
    local result ={}
    for i, v in pairs(tbl) do
        pdtype = pandoc.utils.type(v)   
        if pdtype == "Inlines" or pdtype =="boolean"
        then 
            result[i] = str(v)
        elseif pdtype == "List" then result[i] = DeInline(v)
        end
    end  
    return(result)
end

local function tablecontains(tbl, val)
  if tbl ~= nil and val ~= nil then
    for _, v in ipairs(tbl) do
      if val == v then return true end
    end
  end
  return false
end  

local function ifelse(condition, iftrue, iffalse)
  if condition then return iftrue else return iffalse end
end  

local function replaceifnil(existvalue, replacevalue)
  if existvalue ~= nil then return existvalue else return replacevalue end
end  

local function replaceifempty(existvalue, replacevalue)
  if existvalue == nil or existvalue=="" then return replacevalue else return existvalue end
end  


local function updateTable (oldtbl, newtbl, ignorekeys)
  local result = {}
  -- copy old attributes
  for k, v in pairs(oldtbl) do result[k] = v end
  if newtbl ~= nil then if type(newtbl) == "table" then
      if newtbl[1] == nil then -- it is an ok table with key value pairs
        for k, v in pairs(newtbl) do
          if not(tablecontains(ignorekeys, k)) then
             result[k] = v
         end
        end
      -- special: set reflabel to label if not given in attribs
--        if newattribs["reflabel"] == nil then result.reflabel = result.label end 
      -- TODO: do this elsewhere
      end  
    end
  end  
  return(result)
end



---- init step ---------------------------
--- init = require ("fbx1")

-- find chapter number and file name
-- returns a table with keyed entries
--   processedfile: string, 
--   ishtmlbook: boolean, 
--   chapno: string (at least if ishtmlbook),
--   unnumbered: boolean - initial state of section / chapter
-- if the user has given a chapno yaml entry, then unnumbered = false

-- !!! for pdf, the workflow is very different! ---
-- also find out if lastfile of a book

-- find first and last file of a book, and chapter number of that file 
local function chapterinfo(book, fname)
  local first = "" 
  local last = "" 
  local chapno = nil
  local info = {}
  --if book.render then
    for _, v in pairs(book.render) do
      if str(v.type) == "chapter" then
        last = pandoc.path.split_extension(str(v.file))
        if first == "" then first = last end
        if last == fname then chapno = v.number end
      end
    end
    info.islast = (fname == last)
    info.isfirst = (fname == first)
    info.lastchapter = last
    info.chapno = chapno
    pout("chapter inf:", info)
    return(info)
end

local function Meta_findChapterNumber(meta)
  local processedfile = pandoc.path.split_extension(PANDOC_STATE.output_file)
  fbx.processedfile = processedfile
  fbx.xreffile ="._"..processedfile.."_xref.json"
  fbx.output_file = PANDOC_STATE.output_file
 -- pout(" now in "..processedfile.." later becomes ".. str(fbx.output_file))
  fbx.ishtmlbook = meta.book ~= nil and not quarto.doc.is_format("pdf")
  fbx.isfirstfile = not fbx.ishtmlbook
  fbx.islastfile = not fbx.ishtmlbook
  if fbx.ishtmlbook then 
    local chinfo = chapterinfo(meta.book, processedfile)
    fbx.xreffile= "._"..chinfo.lastchapter.."_xref.json"
    fbx.isfirstfile = chinfo.isfirst 
    fbx.islastfile  = chinfo.islast 
    
    fbx.unnumbered = false
    -- user set chapter number overrides information from meta
    if meta.chapno then  
      fbx.chapno = str(meta.chapno)
    else
      if chinfo.chapno ~= nil then
        fbx.chapno = str(chinfo.chapno)
      else  
        fbx.chapno = ""
        fbx.unnumbered = true
      end
    end
  else -- not a book. 
    fbx.chapno = ""
    fbx.unnumbered = true
  end
end

local function makeKnownClassDetector(knownclasses)
  return function(div)
    for _, cls in pairs(div.classes) do
      if tablecontains(knownclasses, cls) then return str(cls) end
    end
    return nil  
  end
end  

local function Meta_initClassDefaults (meta) 
  -- do we want to prefix fbx numbers with section numbers?
  local cunumbl = meta["custom-numbered-blocks"]
  fbx.knownclasses = {}
  fbx.lists = {}
  --[[ TODO later
  if meta.fbx_number_within_sections then
    fbx.number_within_sections = meta.fbx_number_within_sections
  else   
    fbx.number_within_sections = false
  end 
  --]] 
  -- prepare information for numbering fboxes by class
  -- fbx.knownClasses ={}
  fbx.classDefaults ={}
  local groupDefaults = {default = stylez.defaultOptions} -- not needed later
  fbx.counter = {unnumbered = 0} -- counter for unnumbered divs 
  -- ! unnumbered not for classes that have unnumbered as default !
  -- fbx.counterx = {}
  if cunumbl.classes == nil then
        print("== @%!& == Warning == &!%@ ==\n wrong format for fboxes yaml: classes needed")
        return     
  end
  
-- simplified copy of yaml data: inlines to string
  if cunumbl.groups then
    for key, val in pairs(cunumbl.groups) do
      local ginfo = DeInline(val)
      --[[
      pout("==== before after =======");  pout(ginfo)
      if ginfo.boxstyle then
        local mainstyle, substyle = ginfo.boxstyle:match "([^.]*).(.*)"
      --  pout("main "..mainstyle.." :: "..substyle)
        -- TODO: here account for multiple styles
      end
      --]]--
      ginfo = updateTable(stylez.defaultOptions, ginfo)
      --fbx.
      groupDefaults[key] = ginfo
   --   pout("-----group---"); pout(ginfo)
    end 
  end  
  for key, val in pairs(cunumbl.classes) do
    local clinfo = DeInline(val)
  --  pout("==== before after =======");  pout(clinfo)
    -- classinfo[key] = DeInline(val)
    table.insert(fbx.knownclasses, str(key))
    local theGroup = replaceifnil(clinfo.group, "default")
    clinfo = updateTable(groupDefaults[theGroup], clinfo)
    clinfo.label = replaceifnil(clinfo.label, str(key))
    clinfo.reflabel = replaceifnil(clinfo.reflabel, clinfo.label)
    -- assign counter --  
    clinfo.cntname = replaceifnil(clinfo.group, str(key))
    fbx.counter[clinfo.cntname] = 0 -- sets the counter up if non existing
    fbx.classDefaults[key] = clinfo
 -- pout("---class----");  pout(clinfo)
  end 
  fbx.is_cunumblo = makeKnownClassDetector(fbx.knownclasses)
-- gather lists-of and make filenames by going through all classes
  for _, val in pairs(fbx.classDefaults) do
  --  pout("--classdefault: "..str(key))
  --  pout(val)
    if val.listin then
      for _,v in ipairs(val.listin) do
        fbx.lists[v] = {file = "list-of-"..str(v)..".qmd"}
      end
    end
  end
-- initialize lists
  for key, val in pairs(fbx.lists) do
    val.contents = ifelse(fbx.isfirstfile, "\\providecommand{\\Pageref}[1]{\\hfill p.\\pageref{#1}}", "")
  -- listin approach does not require knownclass, since listin is in classdefaults
  end
 -- pout(fbx.lists)
  --]]
-- document can give the chapter number for books in yaml header 
-- this becomes the counter Prefix
end
 
local initMeta = function(m)
  if m["custom-numbered-blocks"] then
    Meta_findChapterNumber(m)
    Meta_initClassDefaults(m)
  else
    print("== @%!& == Warning == &!%@ ==\n missing cunumblo key in yaml")  
  end
  return(m)
end

----------------------- oldcode, mostly -------------------------------------


------- numbering and class attributes ------

local function fboxDiv_setAttributes(el, cls, prefix)
  local ela = el.attributes -- shortcut
  local ClassDef = fbx.classDefaults[cls]
  --local unnumbered = ClassDef.numbered == "false"
  local numbered = ClassDef.numbered ~= "false"
  local tag = ela.tag
  local tagged = tag ~= nil
  local id = el.identifier
  local autoid =""
--  local titl = ela.title
  local cntkey = ClassDef.cntname
  local counter = {}
  local cnts = 0
  local idnumber = "0.0"
   
  --  set prefix
  ela._prefix = prefix

  id = replaceifnil(id ,"") 
  tag = replaceifnil(tag ,"") 
  
  --- determine if numbered and / or tagged ------
  
  if tagged then numbered = false end
  if el.classes:includes("unnumbered") then numbered = false end

  if ela.numtag then 
    tag = ela.numtag
  --  print("!!! also hier mal ein numtag.\n")
    numbered = true
    tagged = true
  end

-- make counts ---  
  
  if not numbered then cntkey = "unnumbered" end
    
  cnts = fbx.counter[cntkey] +1
  fbx.counter[cntkey] = cnts
 
  idnumber = ifelse(prefix ~= "", prefix .. '.' .. cnts, str(cnts))
  --[[
  if prefix ~="" then  idnumber = prefix .. '.' .. cnts
    else idnumber = str(cnts)
  end    

  if numbered then 
    if  not tagged then tag = idnumber
    else tag = idnumber.."("..tag..")" 
    end  
  end  
]]--
  if numbered then tag = idnumber..ifelse(tagged, "("..tag..")", "" ) end

  if id == "" then
    if numbered then
      autoid = ela._fbxclass..'-'..tag
    else
      autoid = ela._fbxclass..'*-'..idnumber
    end  
    -- changed my mind here: always give autoid
    else autoid = ela._fbxclass..'-id-'..id
  end  
  
  -- do not change identifier el.identifier = id
  
  ela._autoid = autoid 
   
  ela._tag = tag

  ela._file = fbx.processedfile -- necessary to reference between chapters. At least with  quarto 1.3
 -- pout("tag: "..tag)
 -- pout(ela)
  return(el)
end

-- initial attributes without prefix and counts to allow for inner boxes

local function fboxDiv_mark_for_processing(div)
  local diva=div.attributes
  local cls = fbx.is_cunumblo(div)
  local ClassDef = fbx.classDefaults[cls]
  if(cls) then
    diva._process_me = "true"
    diva._fbxclass = str(cls)
    diva._prefix = ""
    diva._tag = ""
    diva._collapse = str(replaceifnil(diva.collapse, ClassDef.collapse)) 
    diva._boxstyle = str(replaceifnil(diva.boxstyle, ClassDef.boxstyle)) 
    diva._label = str(replaceifnil(diva.label, ClassDef.label)) 
    diva._reflabel = str(replaceifnil(diva.reflabel, ClassDef.reflabel)) 
  end  
  return(div)
end



local function Pandoc_prefix_count(doc)
  -- do evt later: non numeric chapernumbers
  local secno = 0
  local prefix = "0"
  if fbx.ishtmlbook then prefix = fbx.chapno end
 
-- pout("this is a book?"..str(fbx.ishtmlbook))

 --- make numbering and prep div blocks ---
--[[------- comment -----------
 quarto (1.2) books allow level 1 headings within a chapter. 
  This would give a mess for crossreference numbers: e.g. multiple examples 3.1,
  from chapter 1 (with 2 l1 headers ) and chapter 3.
 Therefore I decided to ignore level 1 headings in chapters.
 This can easily be changed, then the crossref is for the last occurence only.
 Maybe one day when there is more fine tuning concerning individual numbering depth.
 If this happens before quarto 1.4
   
--]]---------- end comment ------------  
  for i, blk in ipairs(doc.blocks) do
--    print(prefix.."-"..i.." "..blk.t.."\n")
  
    if blk.t=="Header" and not fbx.ishtmlbook then 
      if (blk.level == 1) then -- increase prefix
         if  blk.classes:includes("unnumbered") 
         then 
           prefix = "" 
         else
            secno = secno + 1
            prefix = str(secno)
         end
         -- reset counters in fbx --
         -- this would be more complicated if there are different levels
         -- of numbering depth
         -- then: add a numdepth variable to fbx with a list of keys
         for k in pairs(fbx.counter) do fbx.counter[k]=0 end
      end  

      -- problem: only the outer divs are captured
    elseif blk.t=="Div" then 
        local known = fbx.is_cunumblo(blk)
        if  known then 
           blk = fboxDiv_setAttributes(blk, known, prefix)
        end  
    end
  end  
  return(doc)
end

-- if no id, get from first included header, if possible
local function Divs_getid(el)
  -- local known = getKnownEnv(el.attr.classes)
   local ela = el.attributes
   local id = el.identifier
   
   if not ela._process_me then return(el) end
   -- pout("--- processing item with id ".. replaceifempty(id, "LEER"))
   
   if id == nil or id =="" then  
    -- try in next header
    el1 = el.content[1]
    if el1.t=="Header" then 
    --    pout("--- looking at header with id "..el1.identifier)
    --    pout("--- still processing item with id ".. replaceifempty(id, "LEER"))
     -- pout("replacing id")
      id = el1.identifier
      el.identifier = id
    end
  end 
  if id == nil or id =="" 
    then  
      -- pout("immer noch leer")
      if ela._autoid ~= nil then
      id = ela._autoid
      el.identifier = id
    end 
    --else pout("nix autoid in ");pout(ela._autoid)
  end
  -- pout("resulting el:"); pout(el.attr)
  return(el)
end


--- utility function: stringify and sanitize math, depends on format ---
local function str_sanimath(theInline, fmt)
  local newstring = theInline:walk{
    Math = function(ma)
      local mathtxt = str(ma.text)
      if fmt == "html" then 
        return {'<span class="math inline">\\(' .. mathtxt .. '\\)</span>'}
      elseif fmt == "pdf" then 
        return {'\\(' .. mathtxt .. '\\)'}
      elseif fmt == "md" then 
        return  {'$' .. mathtxt .. '$'}
      else return {mathtxt}
      end  
  end  
  }
  return str(newstring)
end  


----------- title of divs ----------------
local function Divs_maketitle(el)
  -- local known = getKnownEnv(el.attr.classes)
   local ela = el.attributes
   local titl = ela.title
   local mdtitl = replaceifnil(ela.title, "")
   local ClassDef = {}
   -- local id = el.identifier
   
   if not ela._process_me then return(el) end
  -- pout("--- processing item with id ".. replaceifempty(el.identifier, "LEER"))
   
   ClassDef = fbx.classDefaults[ela._fbxclass]
 
   if titl == nil then  
      el1 = el.content[1]
      if el1.t=="Header" then 
        -- sanitize math inline. depends on format
        ela.title = str(el1.content)  -- readable version without math
        mdtitl = str_sanimath(el1.content, "md")
        if ishtml then titl = str_sanimath(el1.content, "html")
          elseif ispdf then titl = str_sanimath(el1.content, "pdf")
          else titl = mdtitl
        end 
    --    pout("--- looking at header with id "..el1.identifier)
    --    pout("--- still processing item with id ".. replaceifempty(id, "LEER"))
    --[[    
    if id =="" or id == nil then
            pout("replacing id")
            id = el1.identifier
            el.identifier = id
        end  
        ]]--
        table.remove(el.content, 1)
      else titl = ""
      end
   end
   ela._title = titl    -- keep the title as attribute for pandoc
   ela._mdtitle = mdtitl    -- for list of
   -- replace empty identifier with autoid
   -- if el.identifier == "" then el.identifier = ela._autoid end
   -- pout("--> sanitarer titel: "..mdtitl)
  -- ela._tag = ""
  -- pout("resulting el:"); pout(el)
   return(el)
 end
 
---------------- initialize xref ----------
-- xrefinit = require ("fbx3")

-- xref split into prepare and finalize to allow xref in titles (future)
local function Pandoc_preparexref(doc)
 -- local xref={}
  local id = ""
  local cnt = 0
  local bla={}
  local xinfo={}
  local file_autoid = {}
  local exists = false
  if fbx.xref == nil then fbx.xref ={} end
  xref = fbx.xref
  cnt = #xref  
  if cnt > 0 then
    for i, xinf in ipairs(xref) do
      file_autoid[xinf.file..xinf.autoid] = i
    end  
  -- pout(autoids)  
  end  
    for _, blk in ipairs(doc.blocks) do
      if blk.attributes then
        bla = blk.attributes
        if bla._process_me then 
          --pout("fbox "..blk.attributes._tag) 
          ------------- an fbox :) ------------
          if blk.identifier == nil then id = ""
            else id = blk.identifier end
          xinfo = {
            id       = id,
            autoid   = bla._autoid,
            cls      = bla._fbxclass,
            label    = bla._label,
            reflabel = bla._reflabel,
            reftag   = bla._tag,
            refnum   = replaceifempty(bla._tag, "??"), 
            file     = fbx.output_file
          }
          -- if not xinfo.reftag then xinfo.reftag ="" end
          -- if xinfo.refnum == ""  then xinfo.refnum ="??" end
          --[[
          if bla._tag 
              then 
                if bla._tag ~="" then xinfo.refnum = bla._tag else xinfo.reftag="??" end
              end 
           ]]--   
          --- check if autoid already exist in database. otherwise update 
          oldxrefno = file_autoid[xinfo.file..xinfo.autoid]
          if oldxrefno == nil then         
            cnt = cnt+1
            bla._xrefno = cnt
            table.insert (xref, cnt, xinfo)
          else
            bla._xrefno = oldxrefno
            xref[oldxrefno] = xinfo
          end
        end 
      end   
    end
    return(doc)
  end  
  
  
local function Pandoc_finalizexref(doc)
  xref = fbx.xref -- shortcut
  local bla = {}
   -- pout("------- finale ----")
    for i, blk in ipairs(doc.blocks) do
      bla = blk.attributes
      --pout(bla)
      if bla then
        if bla._process_me == "true" then 
          ------------- an fbox :) ------------
          xindex = tonumber(bla._xrefno)
          if xindex then
            xref[xindex].neu = true -- mark as new entry
            if bla._title then xref[xindex].title = bla._title end
          end
         
        --  else pout("ochje.")
        end  
      end   
    end
 -- pout(xref)
  --- write to disc --
  --- check if this was the last file to process ---
  return(doc)
end  


local function Meta_writexref(meta)
  local xref = fbx.xref
  local xrjson = quarto.json.encode(fbx.xref)
  local file = io.open(fbx.xreffile,"w")
  if file ~= nil then 
    file:write(xrjson) 
    file:close()
  end
  if fbx.islastfile then 
  --  pout(fbx.processedfile.." -- nu aufräum! aber zack ---") 
    for i, v in ipairs(xref) do
      if not v.neu then 
  --      pout("killed")
  --      pout(v)
        xref[i] = nil
      end   
    end
  --  pout("-------- überlebende")
  --  pout(xref)
  end  
end

  
local function Meta_readxref(meta)
  local file = io.open(fbx.xreffile,"r")
  if file then 
     local xrfjson = file:read "*a"
    file:close()
    --[[
    if xrfjson then meta.fbx.xref = quarto.json.decode(xrfjson)
    else meta.fbx.xref = {} end
   --   pout("eingelesen")
   -- pout(meta.fbx.xref)
  else meta.fbx.xref ={}
    --]]--
    if xrfjson then fbx.xref = quarto.json.decode(xrfjson)
    else fbx.xref = {} end
    --  pout("eingelesen")
    --pout(fbx.xref)
  else fbx.xref ={}
  end  
  return(meta)
end
-------------- render -------------------
-- render = require ("fbx4")

local tt_from_attributes_id = function(A, id)
  --local tyti =""
  --local tt = {}
  --if A._tag == "" then tyti = A._label 
  --else tyti = A._label..' '..A._tag end 
--    print("TYTI: === "..tyti)
local thelink = "#"..id
      if fbx.ishtmlbook and A._file~=nil then thelink = A._file..".qmd"..thelink end
return {id = id,
      type = A._fbxclass, 
      tag = A._tag,
      title = A._title, 
      typlabel = A._label,
      typlabelTag = A._label .. ifelse(A._tag == "",""," "..A._tag),
      mdtitle = A._mdtitle, 
      collapse = A._collapse,
      boxstyle = A._boxstyle,
      link = thelink
}
  -- pout("====nun====");pout(tt)
  --return(tt)
end

insertStylesPandoc = function(doc)
  -- if stylez.extractStyleFromYaml then stylez.extractStyleFromYaml() end
  if stylez.insertPreamble and (quarto.doc.is_format("html") or quarto.doc.is_format("pdf"))
    then stylez.insertPreamble(doc, fbx.classDefaults, fmt) end
  return(doc)
end;

renderDiv = function(thediv)    
  local A = thediv.attributes
  local tt = {}
  if A._fbxclass ~= nil then
    
    collapsstr = str(A._collapse)
    tt = tt_from_attributes_id(A, thediv.identifier)
    
    local fmt='html'
    if quarto.doc.is_format("pdf") then fmt = "tex" end;
    if #thediv.content > 0 and thediv.content[1].t == "Para" and 
      thediv.content[#thediv.content].t == "Para" then
        table.insert(thediv.content[1].content, 1, 
          pandoc.RawInline(fmt, stylez.blockStart(tt, fmt)))
        table.insert(thediv.content,
          pandoc.RawInline(fmt, stylez.blockEnd(tt, fmt)))
      else
        table.insert(thediv.content, 1, 
          pandoc.RawBlock(fmt, stylez.blockStart(tt, fmt)))
        table.insert(thediv.content,  
          pandoc.RawBlock(fmt, stylez.blockEnd(tt, fmt)))
    end  
    --]]
  end  
  return(thediv)
end -- function renderDiv


------------- xrefs
-- learned from nameref extension by shafayeedShafee
-- TODO: make everything with walk. Looks so nice
local function resolveref(data)
  return { 
   RawInline = function(el)
      local refid = el.text:match("\\ref{(.*)}")
      if refid then 
        if data[refid] then
          local href = '#'..refid
          if fbx.ishtmlbook then 
            href = data[refid].file .. href 
          end  
          return pandoc.Link(data[refid].refnum, href)
      end  end
    end
  }
end

-- TODO: with filenames for books

function Pandoc_resolvexref(doc)
  local xrefdata = {}
  local xref = fbx.xref
  for _, xinf in pairs(xref) do
    if xinf.id then if xinf.id ~= "" then
      xrefdata[xinf.id] = xinf
  end  end
end
-- pout(xrefdata)
  return doc:walk(resolveref(xrefdata))
end  
-----------

--- remove all attributes that start with underscore. 
-- could theoretically give clashes with filters that need persistent such attributes
function Div_cleanupAttribs (el)
  if el.attributes._process_me then
    for k, v in pairs(el.attributes) do
      if string.sub(k, 1, 1) =="_" then el.attributes[k] = nil end
    end
  end
  return el
end

-- debugging stuff
--[[

local function pandocblocks(doc)
  for k,v in pairs(doc.blocks) do
    pout(v.t)
  end
end

local function pandocdivs(div)
  pout(div.t.." - "..div.identifier)
  pout(div.attributes)
end
]]--

local function Pandoc_makeListof(doc)
  local tt = {}
  local thelists = {}
  local zeile = ""
  local lstfilemode = ifelse(fbx.isfirstfile, "w", "a")
  if not fbx.lists then return(doc) end
  for i, blk in ipairs(doc.blocks) do
    --[[ -- this may require manual deletion of headers in the list-of.qmd
    -- and does in this form not help with html books anyway --
    if blk.t=="Header" then 
      if blk.level==1 then 
        zeile = "\n\n## "..str(blk.content).."\n"
      --- add to all lists
        for _, lst in pairs (fbx.lists) do
          lst.contents = lst.contents..zeile
        end
      end
      elseif blk.t=="Div" then 
     ]]-- 
     if blk.t=="Div" then 
      if blk.attributes._process_me then
        thelists = fbx.classDefaults[blk.attributes._fbxclass].listin
        if thelists ~= nil and thelists ~="" then
          tt = tt_from_attributes_id (blk.attributes, blk.identifier)
          -- pout("thett------");pout(tt)
        --  zeile = ("\n[**"..tt.typtitel.."**](#"..blk.identifier..")"..ifelse(tt.mdtitle=="","",": "..tt.mdtitle)..
        --      " \\Pageref{"..blk.identifier.."}\n")
        -- TODO: should be like [**R-tip 1.1**](intro.qmd#Rtip-install)
          --zeile = ("\n[**"..tt.titeltyp.." \\ref{"..blk.identifier.."}**]" ..
            --       ifelse(tt.mdtitle=="","",": "..tt.mdtitle) ..
              --     " \\Pageref{"..blk.identifier.."}\n") 
            zeile = ("\n [**"..tt.typlabelTag.."**](" .. tt.link ..")" ..
                     ifelse(tt.mdtitle=="","",": "..tt.mdtitle) .. "\\Pageref{".. tt.id .."}\n")                               
          for _, lst in ipairs (thelists) do
            fbx.lists[lst].contents = fbx.lists[lst].contents..zeile
          end 
        end
      end 
    end
  end
  --- write to file ---
  for nam, lst in pairs(fbx.lists) do
    if lst.file ~= nil then 
      local file = io.open(lst.file, lstfilemode)
      if file then 
        file:write(lst.contents) 
        file:close()
      else pout("cannot write to file "..lst.file)  
      end  
    end  
  end
  return(doc)
end

return{
    {Meta = initMeta}
    --[[
    ,{Pandoc = function(d)
      for k, v in pairs(stylez) do
        pout(k..":  ".. type(v))
      end 
    end 
    }
    ]]--
  , {Meta = Meta_readxref, Div=fboxDiv_mark_for_processing,
     Pandoc = Pandoc_prefix_count} 
 -- , {Div=pandocdivs, Pandoc=pandocblocks}
  --[[ ]]
   
  , {Div = Divs_getid, Pandoc = Pandoc_preparexref}
  , {Pandoc = Pandoc_resolvexref}
  , {Div = Divs_maketitle}
  , {Pandoc = Pandoc_finalizexref}
  , {Meta = Meta_writexref, Pandoc = Pandoc_makeListof}
  , {Div = renderDiv}
  , {Pandoc = insertStylesPandoc}
  , {Div = Div_cleanupAttribs}
--[[
  
  -- ]]
}


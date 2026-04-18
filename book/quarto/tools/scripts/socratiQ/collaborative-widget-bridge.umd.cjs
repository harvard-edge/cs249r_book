(function(p,w){typeof exports=="object"&&typeof module<"u"?w(exports):typeof define=="function"&&define.amd?define(["exports"],w):(p=typeof globalThis<"u"?globalThis:p||self,w(p.CollaborativeWidget={}))})(this,function(p){"use strict";class w{constructor(e={}){const o=window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1"||window.location.hostname==="0.0.0.0"?"ws://localhost:8789":"wss://sync-server.mlsysbook.workers.dev";this.signalingServerUrl=e.signalingServerUrl||o,this.iceServers=e.iceServers||[{urls:"stun:stun.l.google.com:19302"},{urls:"stun:stun1.l.google.com:19302"}],this.ws=null,this.myUsername=e.username||this.generateRandomUsername(),this.roomName=null,this.actualRoomName=null,this.myClientId=null,this.isHub=!1,this.peers={},this.typingTimers={},this.typingUsers=new Set,this.userCountValue=1,this.heartbeatInterval=null,this.inputTimeout=null,this.connectionHealth={startTime:Date.now(),lastHeartbeat:Date.now(),connectionAttempts:0,warnings:[],isHealthy:!0},this.quizRoomManager=null,this.recentMessages=new Map,this.commentRetryQueue=[],this.maxRetryAttempts=3,this.retryDelay=2e3,this.onConnectionStatusChange=e.onConnectionStatusChange||(()=>{}),this.onUserCountChange=e.onUserCountChange||(()=>{}),this.onMessage=e.onMessage||(()=>{}),this.onInfoMessage=e.onInfoMessage||(()=>{}),this.onTypingIndicatorChange=e.onTypingIndicatorChange||(()=>{}),this.onUserJoined=e.onUserJoined||(()=>{}),this.onUserLeft=e.onUserLeft||(()=>{}),this.onCommentAdd=e.onCommentAdd||(()=>{}),this.onCommentDelete=e.onCommentDelete||(()=>{}),this.onCommentResolve=e.onCommentResolve||(()=>{}),this.onCommentStatusRequest=e.onCommentStatusRequest||(()=>{}),this.onCommentStatusResponse=e.onCommentStatusResponse||(()=>{}),this.onSystemMessage=e.onSystemMessage||(()=>{}),this.onCursorUpdate=e.onCursorUpdate||(()=>{}),this.onPresenceUpdate=e.onPresenceUpdate||(()=>{}),e.username||(this.myUsername=this.generateRandomUsername()),this.currentPageData=this.extractCurrentPageDataSync()}generateRandomUsername(){const e=["Happy","Cool","Smart","Fast","Bright","Kind","Bold","Wise","Swift","Brave"],t=["Tiger","Eagle","Wolf","Bear","Fox","Lion","Hawk","Falcon","Panther","Lynx"];return`${e[Math.floor(Math.random()*e.length)]}${t[Math.floor(Math.random()*t.length)]}`}extractCurrentPageDataSync(){try{const e=document.body.textContent||"",t=e.split(/\s+/).filter(o=>o.length>0).length;return{text:e||"No content available",url:window.location.href,title:document.title||"Untitled Page",timestamp:Date.now(),wordCount:t,charCount:e.length}}catch(e){return console.warn("Failed to extract page content, using minimal fallback:",e),{text:"No content available",url:window.location.href,title:document.title||"Untitled Page",timestamp:Date.now(),wordCount:0,charCount:0}}}async extractCurrentPageData(){try{const{TextExtractor:e}=await Promise.resolve().then(()=>q),t=e.extractPageText(!0);return{text:t.text||document.body.textContent||"No content available",url:window.location.href,title:document.title||"Untitled Page",timestamp:Date.now(),wordCount:t.wordCount||0,charCount:t.charCount||0}}catch(e){return console.warn("Failed to extract page content with TextExtractor, using fallback:",e),{text:document.body.textContent||"No content available",url:window.location.href,title:document.title||"Untitled Page",timestamp:Date.now(),wordCount:0,charCount:0}}}async updateCurrentPageData(){this.currentPageData=await this.extractCurrentPageData(),console.log("📄 Updated current page data:",{url:this.currentPageData.url,title:this.currentPageData.title,charCount:this.currentPageData.charCount,wordCount:this.currentPageData.wordCount})}setupContentUpdateDetection(){try{Promise.resolve().then(()=>q).then(({TextExtractor:e})=>{e.setupNavigationDetection(t=>{console.log("📄 Page content changed, updating currentPageData"),this.currentPageData={text:t.text,url:t.url,title:t.title,timestamp:Date.now(),wordCount:t.wordCount||0,charCount:t.charCount||0}})})}catch(e){console.warn("Failed to set up automatic content updates:",e)}}async joinRoom(e,t,o=null,s="join",n=!1){this.myUsername=e,this.roomName=t,o?this.actualRoomName=btoa(t+":"+o).replace(/[^a-zA-Z0-9]/g,""):this.actualRoomName=t,n||this.onInfoMessage(`You have ${s.toLowerCase()}ed the room.`,"success"),await this.connectToSignalingServer(s),await this.initializeQuizRoom(),this.setupContentUpdateDetection()}sendMessage(e){if(e){if(Object.values(this.peers).length===0){this.onMessage(this.myUsername,e,!0);return}this.broadcastMessage({type:"chat",payload:e})}}sendTypingIndicator(){Object.values(this.peers).length!==0&&(this.inputTimeout&&clearTimeout(this.inputTimeout),this.broadcastMessage({type:"typing"}),this.inputTimeout=setTimeout(()=>{},500))}disconnect(){this.ws&&this.ws.close(),this.clearRetryQueue(),this.heartbeatInterval&&(clearInterval(this.heartbeatInterval),this.heartbeatInterval=null),this.healthCheckInterval&&(clearInterval(this.healthCheckInterval),this.healthCheckInterval=null),this.inputTimeout&&(clearTimeout(this.inputTimeout),this.inputTimeout=null),Object.values(this.typingTimers).forEach(e=>{clearTimeout(e)}),this.typingTimers={},this.typingUsers.clear(),Object.values(this.peers).forEach(e=>{e.pc.close()}),this.peers={},this.quizRoomManager&&(this.quizRoomManager.stopPeriodicCollection(),this.quizRoomManager=null),this.onConnectionStatusChange(!1)}async connectToSignalingServer(e){const t=`${this.signalingServerUrl}/${this.actualRoomName}/websocket`;this.ws=new WebSocket(t),this.ws.onopen=()=>{console.log(`Connected to signaling server (${e} mode)`),this.onConnectionStatusChange(!0),this.heartbeatInterval=setInterval(()=>{this.ws.readyState===WebSocket.OPEN&&this.ws.send(JSON.stringify({type:"ping"}))},1e4),this.healthCheckInterval=setInterval(()=>{this.monitorConnectionHealth()},15e3)},this.ws.onmessage=async o=>{const s=JSON.parse(o.data);console.log("Signaling message received:",s),await this.handleSignalingMessage(s)},this.ws.onerror=o=>{console.error("WebSocket Error:",o),this.onConnectionStatusChange(!1),this.onInfoMessage("Error connecting to the server.","warning")},this.ws.onclose=()=>{this.onConnectionStatusChange(!1),this.onInfoMessage("Connection to server lost.","warning"),this.heartbeatInterval&&(clearInterval(this.heartbeatInterval),this.heartbeatInterval=null),this.healthCheckInterval&&(clearInterval(this.healthCheckInterval),this.healthCheckInterval=null)}}async handleSignalingMessage(e){const{type:t,peerId:o,from:s,sdp:n,candidate:i,username:a,userCount:r,clientId:c,existingPeers:l}=e;switch(this.detectConnectionIssues(e),t){case"pong":console.log("Received pong from server"),this.connectionHealth.lastHeartbeat=Date.now();break;case"room-info":this.myClientId=c,this.userCountValue=r,this.onUserCountChange(this.userCountValue),console.log(`Joined room with ${r} users`),console.log("Existing peers:",l),r===1&&l.length===0?(this.isHub=!0,this.onInfoMessage(`You created a new room "${this.roomName}" (You are the hub)`,"success"),console.log("🏠 You are the first user in this room - you are the hub"),this.detectPotentialRaceCondition(r,l),this.logConnectionStates()):(this.isHub=!1,this.onInfoMessage(`You joined an existing room "${this.roomName}" with ${r} users`,"success"),console.log("🏠 You joined an existing room - you are NOT the hub"),console.log(`Creating connections to existing peers: ${l.join(", ")}`),console.log(`My clientId: ${this.myClientId}`),l.forEach(d=>{console.log(`Creating connection to existing peer: ${d}`),this.myClientId<d?(console.log(`🔄 Fallback: Both users are non-hub, ${this.myClientId} will initiate connection to existing peer ${d}`),this.createPeerConnection(d,!0,!0)):(console.log(`🔄 Fallback: Both users are non-hub, ${this.myClientId} will wait for existing peer ${d} to initiate`),this.createPeerConnection(d,!1,!0))}),this.logConnectionStates());break;case"user-joined":if(this.userCountValue++,this.onUserCountChange(this.userCountValue),console.log(`User ${o} joined`),this.onUserJoined(o),this.updateRoomUser(o,`User_${o}`,"join"),this.isHub)console.log(`🏠 Hub connecting to new user: ${o}`),this.createPeerConnection(o,!0),this.logConnectionStates();else if(console.log(`🏠 Non-hub user - hub will handle connection to ${o}`),o!==this.myClientId&&(this.myClientId<o?(console.log(`🔄 Fallback: Both users are non-hub, ${this.myClientId} will initiate connection to ${o}`),this.createPeerConnection(o,!0,!0)):(console.log(`🔄 Fallback: Both users are non-hub, ${this.myClientId} will wait for ${o} to initiate`),this.createPeerConnection(o,!1,!0)),this.logConnectionStates()),o===this.myClientId){console.log("🔄 Auto-reconnect detected - waiting for hub to initiate connection");const d=Object.keys(this.peers)[0];d&&!this.peers[d]&&(console.log(`Creating connection to hub: ${d} (waiting for hub to initiate)`),this.createPeerConnection(d,!1)),this.logConnectionStates()}break;case"offer":console.log(`Received offer from ${s}`),await this.handleOffer(s,n);break;case"answer":console.log(`Received answer from ${s}`),await this.handleAnswer(s,n);break;case"ice-candidate":console.log(`Received ICE candidate from ${s}`),await this.handleIceCandidate(s,i);break;case"user-left":console.log(`User ${o} left`),this.updateRoomUser(o,`User_${o}`,"leave"),this.handleUserLeft(o);break}}createPeerConnection(e,t,o=!1){if(this.peers[e]){console.warn(`Connection to ${e} already exists or is in progress.`);return}t&&!this.isHub&&!o?(console.warn(`⚠️ Non-hub user attempting to initiate connection to ${e} - this could cause dual-offer conflicts`),t=!1,console.log(`🔄 Converted to receiver mode for ${e} to prevent dual-offer`)):t&&!this.isHub&&o&&console.log(`✅ Fallback-initiated connection allowed: ${this.myClientId} initiating to ${e}`),console.log(`Creating peer connection to ${e}, isInitiator: ${t}, isHub: ${this.isHub}`);const s=new RTCPeerConnection({iceServers:this.iceServers});if(this.peers[e]={pc:s,username:"...joining...",isInitiator:t,connectionState:"creating",pendingIceCandidates:[]},console.log("Peer connection created, total peers:",Object.keys(this.peers).length),s.onicecandidate=n=>{n.candidate&&this.ws.send(JSON.stringify({type:"ice-candidate",to:e,candidate:n.candidate}))},s.onconnectionstatechange=()=>{console.log(`🔗 Connection state with ${e}: ${s.connectionState}`),this.peers[e].connectionState=s.connectionState,s.connectionState==="connected"?console.log(`✅ Successfully connected to ${e}`):s.connectionState==="disconnected"||s.connectionState==="failed"||s.connectionState==="closed"?(console.log(`❌ Connection lost with ${e}`),this.handleUserLeft(e)):s.connectionState==="connecting"&&setTimeout(()=>{this.peers[e]&&this.peers[e].connectionState==="connecting"&&(console.log(`⚠️ Connection to ${e} is stuck in 'connecting' state - this may indicate a deadlock`),console.log("🔍 Current connection states:",Object.entries(this.peers).map(([n,i])=>`${n}: ${i.connectionState}`)))},1e4)},s.oniceconnectionstatechange=()=>{console.log(`🧊 ICE connection state with ${e}: ${s.iceConnectionState}`)},t){console.log(`🚀 Creating data channel for ${e} (Hub: ${this.isHub})`);const n=s.createDataChannel("chat",{ordered:!0,maxRetransmits:3});this.setupDataChannel(n,e),this.peers[e].dc=n,s.createOffer().then(i=>(console.log(`📤 Created offer for ${e} (Hub: ${this.isHub})`),s.setLocalDescription(i))).then(()=>{console.log(`📤 Sending offer to ${e} (Hub: ${this.isHub})`),this.ws.send(JSON.stringify({type:"offer",to:e,sdp:s.localDescription}))}).catch(i=>console.error(`❌ Error creating offer for ${e}:`,i))}else console.log(`⏳ Waiting for data channel from ${e} (Hub: ${this.isHub})`),s.ondatachannel=n=>{console.log(`📨 Received data channel from ${e} (Hub: ${this.isHub})`);const i=n.channel;this.setupDataChannel(i,e),this.peers[e].dc=i}}setupDataChannel(e,t){e.onopen=()=>{console.log(`✅ Data channel with ${t} is open!`),e.send(JSON.stringify({type:"username",payload:this.myUsername,url:window.location.href,title:document.title})),console.log(`Sent username to ${t}: ${this.myUsername}`),this.triggerRetryQueue()},e.onmessage=o=>{const s=JSON.parse(o.data);s.type!=="cursor_update"&&s.type!=="presence_update"&&console.log(`📨 Received message from ${t}:`,s),this.handleDataChannelMessage(s,t)},e.onclose=()=>{console.log(`❌ Data channel with ${t} has closed.`),this.handleUserLeft(t)},e.onerror=o=>{console.error(`❌ Data channel error with ${t}:`,o)}}handleDataChannelMessage(e,t){const o=this.peers[t]?.username||"...";switch(e.type){case"chat":this.onMessage(o,e.payload,!1);break;case"ai_response":this.onMessage(o,e,!1);break;case"quiz_submission":this.onMessage(o,e,!1);break;case"quiz_lock":this.onMessage(o,e,!1);break;case"countdown_start":console.log(`🎮 Received countdown start from ${o}:`,e.payload),this.onMessage(o,e,!1);break;case"typing":this.showTypingIndicator(o);break;case"username":this.peers[t].username=e.payload,this.peers[t].url=e.url,this.peers[t].title=e.title,this.onInfoMessage(`${e.payload} has connected.`,"success");break;case"comment_add":this.onCommentAdd(e.data,t);break;case"comment_delete":this.onCommentDelete(e.data,t);break;case"comment_resolve":console.log("Received comment_resolve message:",e.data),this.onCommentResolve(e.data.commentId,e.data.highlightId,e.data.url,e.data.room,e.data.resolvedBy);break;case"system_message":this.onSystemMessage(e.payload,t);break;case"cursor_update":this.onCursorUpdate(e,t);break;case"presence_update":this.onPresenceUpdate(e,t);break;case"comment_status_request":this.onCommentStatusRequest(e.data,t);break;case"comment_status_response":this.onCommentStatusResponse(e.data,t);break;case"collect_content":case"content_response":case"quiz_message":case"leader_selected":case"quiz_answer":this.handleAIMessage(e,t);break}}async handleOffer(e,t){console.log(`📥 Received offer from ${e}`),this.peers[e]||(console.log(`🆕 Creating new peer connection for ${e}`),this.createPeerConnection(e,!1));const o=this.peers[e].pc;if(o.signalingState==="stable"&&this.peers[e].connectionState==="connected"){console.log(`⚠️ Ignoring offer from ${e} - connection already established`);return}if(o.signalingState==="have-local-offer"&&this.peers[e].connectionState==="connected"){console.log(`⚠️ Ignoring offer from ${e} - connection already established (dual-offer prevention)`);return}else o.signalingState==="have-local-offer"&&console.log(`🔄 Processing offer from ${e} despite local offer - connection not yet established`);try{console.log(`📥 Setting remote description for ${e}`),await o.setRemoteDescription(new RTCSessionDescription(t)),await this.processQueuedIceCandidates(e),console.log(`📤 Creating answer for ${e}`);const s=await o.createAnswer();console.log(`📤 Setting local description for ${e}`),await o.setLocalDescription(s),console.log(`📤 Sending answer to ${e}`),this.ws.send(JSON.stringify({type:"answer",to:e,sdp:o.localDescription}))}catch(s){console.error(`❌ Error handling offer from ${e}:`,s)}}async handleAnswer(e,t){if(console.log(`📥 Received answer from ${e}`),this.peers[e])try{if(this.peers[e].pc.signalingState==="stable"){console.log(`⚠️ Ignoring answer from ${e} - connection already stable`);return}console.log(`📥 Setting remote description for ${e}`),await this.peers[e].pc.setRemoteDescription(new RTCSessionDescription(t)),await this.processQueuedIceCandidates(e),console.log(`✅ Set remote description for ${e}`)}catch(o){console.error(`❌ Error handling answer from ${e}:`,o)}else console.error(`❌ No peer connection found for ${e}`)}async handleIceCandidate(e,t){if(console.log(`🧊 Received ICE candidate from ${e}`),this.peers[e])try{this.peers[e].pc.remoteDescription?(await this.peers[e].pc.addIceCandidate(new RTCIceCandidate(t)),console.log(`✅ Added ICE candidate for ${e}`)):(this.peers[e].pendingIceCandidates.push(t),console.log(`⏳ Queued ICE candidate for ${e} (remote description not set yet)`))}catch(o){console.error(`❌ Error adding ICE candidate for ${e}:`,o)}else console.error(`❌ No peer connection found for ${e}`)}async processQueuedIceCandidates(e){if(!this.peers[e]||!this.peers[e].pendingIceCandidates)return;const t=this.peers[e].pendingIceCandidates;if(t.length!==0){console.log(`🔄 Processing ${t.length} queued ICE candidates for ${e}`);for(const o of t)try{await this.peers[e].pc.addIceCandidate(new RTCIceCandidate(o)),console.log(`✅ Added queued ICE candidate for ${e}`)}catch(s){console.error(`❌ Error adding queued ICE candidate for ${e}:`,s)}this.peers[e].pendingIceCandidates=[],console.log(`✅ Processed all queued ICE candidates for ${e}`)}}handleUserLeft(e){if(this.peers[e]){const t=this.peers[e].username;console.log(`👋 User ${t} (${e}) left the room`),this.onUserLeft(e,t),this.hideTypingIndicator(t),this.peers[e].pc.close(),this.peers[e].pendingIceCandidates&&(this.peers[e].pendingIceCandidates=[]),delete this.peers[e],this.userCountValue=Math.max(1,this.userCountValue-1),this.onUserCountChange(this.userCountValue),console.log(`📊 Updated user count: ${this.userCountValue}, remaining peers: ${Object.keys(this.peers).length}`)}else console.log(`👋 User ${e} left before connection established`),this.userCountValue=Math.max(1,this.userCountValue-1),this.onUserCountChange(this.userCountValue)}logConnectionStates(){console.log("🔍 Connection Debug Info:"),console.log(`  - Is Hub: ${this.isHub}`),console.log(`  - My Client ID: ${this.myClientId}`),console.log(`  - Total Peers: ${Object.keys(this.peers).length}`),console.log("  - Peer Details:"),Object.entries(this.peers).forEach(([e,t])=>{console.log(`    * ${e}: ${t.connectionState} (initiator: ${t.isInitiator})`)})}detectConnectionIssues(e){const{type:t,userCount:o,existingPeers:s,clientId:n}=e;this.connectionTiming||(this.connectionTiming={startTime:Date.now(),roomInfoReceived:!1,firstUserJoined:!1,connectionAttempts:0}),t==="room-info"&&(this.connectionTiming.roomInfoReceived=!0,this.connectionTiming.roomInfoTime=Date.now(),this.detectSuspiciousPatterns(o,s,n)),t==="user-joined"&&(this.connectionTiming.firstUserJoined=!0,this.connectionTiming.firstUserJoinedTime=Date.now())}detectSuspiciousPatterns(e,t,o){const s=[];e>1&&t.length===0&&s.push(`🚨 CRITICAL: Server reports ${e} users but no existing peers! This suggests a Durable Object reset/eviction.`);const n=Date.now()-this.connectionTiming.startTime;if(n<100&&s.push(`⚠️ FAST CONNECTION: Room info received in ${n}ms - potential race condition`),o&&this.previousClientIds){const i=Date.now()-(this.previousClientIds.lastSeen||0);i<5e3&&s.push(`⚠️ RAPID RECONNECT: New client ID generated ${i}ms after previous - possible connection issues`)}this.previousClientIds||(this.previousClientIds=[]),this.previousClientIds.push({clientId:o,timestamp:Date.now(),lastSeen:Date.now()}),s.length>0&&(console.group("🚨 CONNECTION ISSUES DETECTED"),s.forEach(i=>console.warn(i)),console.groupEnd(),this.onInfoMessage(`Connection issues detected: ${s.length} potential problems`,"warning"))}detectPotentialRaceCondition(e,t){const o=[],s=Date.now()-this.connectionTiming.startTime;s<200&&o.push(`Hub assigned in ${s}ms (very fast)`),e>1&&t.length===0&&o.push(`Server inconsistency: ${e} users but no existing peers`),this.connectionTiming.connectionAttempts>1&&o.push("Multiple connection attempts detected"),o.length>0&&(console.group("⚠️ POTENTIAL RACE CONDITION DETECTED"),o.forEach(n=>console.warn(`  - ${n}`)),console.groupEnd(),this.onInfoMessage("⚠️ Potential race condition detected. If another user joins and doesn't connect, try refreshing.","warning"))}monitorConnectionHealth(){const e=Date.now(),t=e-this.connectionHealth.startTime,o=e-this.connectionHealth.lastHeartbeat;o>3e4&&(this.connectionHealth.warnings.push(`No heartbeat for ${Math.round(o/1e3)}s`),this.connectionHealth.isHealthy=!1);const s=Object.values(this.peers).filter(n=>n.pc&&n.pc.connectionState==="connected").length;return this.userCountValue>1&&s===0&&(this.connectionHealth.warnings.push(`No WebRTC connections despite ${this.userCountValue} users in room`),this.connectionHealth.isHealthy=!1),this.connectionHealth.isHealthy||(console.group("🏥 CONNECTION HEALTH CHECK"),console.warn(`Connection unhealthy after ${Math.round(t/1e3)}s`),this.connectionHealth.warnings.forEach(n=>console.warn(`  - ${n}`)),console.groupEnd(),this.onInfoMessage("⚠️ Connection issues detected. Consider refreshing if problems persist.","warning")),this.connectionHealth.isHealthy}broadcastMessage(e){const t=JSON.stringify(e);if(e.type==="comment_add"&&console.log("🔍 DEBUG: broadcastMessage called for comment_add:",{peerCount:Object.keys(this.peers).length,peers:Object.keys(this.peers),message:e}),Object.keys(this.peers).length===0){e.type==="chat"&&this.onInfoMessage("No other users in the room to send message to.","warning"),e.type==="comment_add"&&(console.log("⚠️ No peers available for comment broadcast - queuing for retry"),this.queueCommentForRetry(e));return}const o=`${e.type}-${JSON.stringify(e.payload||e)}`,s=Date.now();if(this.recentMessages.has(o)){const n=this.recentMessages.get(o);if(s-n<1e3){console.log("🚫 Duplicate message blocked:",e.type);return}}this.recentMessages.set(o,s);for(const[n,i]of this.recentMessages.entries())s-i>5e3&&this.recentMessages.delete(n);e.type!=="cursor_update"&&e.type!=="presence_update"&&console.log("📤 Broadcasting message:",e),e.type==="chat"?this.onMessage(this.myUsername,e.payload,!0):e.type==="comment_add"?this.onCommentAdd(e.data):e.type==="system_message"&&this.onSystemMessage(e.payload,this.myClientId),Object.entries(this.peers).forEach(([n,i])=>{i.dc&&i.dc.readyState==="open"?(i.dc.send(t),e.type==="comment_add"&&console.log("🔍 DEBUG: Comment message sent to peer:",n)):e.type==="comment_add"&&console.log("⚠️ Comment message not sent - peer data channel not open:",{peerId:n,readyState:i.dc?.readyState})})}queueCommentForRetry(e){const t={message:e,attempts:0,timestamp:Date.now(),id:`retry-${Date.now()}-${Math.random().toString(36).substr(2,9)}`};this.commentRetryQueue.push(t),console.log(`📝 Comment queued for retry: ${t.id}`),this.scheduleRetry()}scheduleRetry(){this.retryTimeout&&clearTimeout(this.retryTimeout),this.retryTimeout=setTimeout(()=>{this.processRetryQueue()},this.retryDelay)}processRetryQueue(){if(this.commentRetryQueue.length===0)return;console.log(`📝 Processing retry queue: ${this.commentRetryQueue.length} comments pending`);const e=Date.now(),t=3e4;this.commentRetryQueue=this.commentRetryQueue.filter(o=>e-o.timestamp>t?(console.log(`📝 Removing expired comment from retry queue: ${o.id}`),!1):o.attempts>=this.maxRetryAttempts?(console.log(`📝 Max retry attempts reached for comment: ${o.id}`),!1):(this.retryComment(o),!0)),this.commentRetryQueue.length>0&&this.scheduleRetry()}retryComment(e){if(e.attempts++,console.log(`📝 Retrying comment ${e.id} (attempt ${e.attempts}/${this.maxRetryAttempts})`),Object.keys(this.peers).length>0){console.log(`📝 Peers available, retrying comment: ${e.id}`);try{const t=JSON.stringify(e.message);Object.entries(this.peers).forEach(([s,n])=>{n.dc&&n.dc.readyState==="open"&&(n.dc.send(t),console.log(`📝 Retry successful - comment sent to peer: ${s}`))});const o=this.commentRetryQueue.findIndex(s=>s.id===e.id);o!==-1&&(this.commentRetryQueue.splice(o,1),console.log(`📝 Comment retry successful, removed from queue: ${e.id}`))}catch(t){console.error(`❌ Failed to retry comment ${e.id}:`,t)}}else console.log(`📝 Still no peers available for comment ${e.id}`)}clearRetryQueue(){console.log(`📝 Clearing retry queue: ${this.commentRetryQueue.length} items`),this.commentRetryQueue=[],this.retryTimeout&&(clearTimeout(this.retryTimeout),this.retryTimeout=null)}triggerRetryQueue(){this.commentRetryQueue.length>0&&(console.log("📝 Peer connected, triggering immediate retry queue processing"),this.retryTimeout&&(clearTimeout(this.retryTimeout),this.retryTimeout=null),this.processRetryQueue())}showTypingIndicator(e){this.typingUsers.add(e),this.typingTimers[e]&&clearTimeout(this.typingTimers[e]),this.typingTimers[e]=setTimeout(()=>{this.hideTypingIndicator(e)},1500),this.updateTypingDisplay()}hideTypingIndicator(e){this.typingUsers.delete(e),this.typingTimers[e]&&(clearTimeout(this.typingTimers[e]),delete this.typingTimers[e]),this.updateTypingDisplay()}updateTypingDisplay(){if(this.typingUsers.size===0){this.onTypingIndicatorChange("",[]);return}const e=Array.from(this.typingUsers);this.typingUsers.size===1?this.onTypingIndicatorChange(e[0],e):this.onTypingIndicatorChange(e.join(", "),e)}getUsername(){return this.myUsername}getRoomName(){return this.roomName}getUserCount(){return this.userCountValue}isConnected(){return this.ws&&this.ws.readyState===WebSocket.OPEN}isHubUser(){return this.isHub}async initializeQuizRoom(){try{const{QuizRoomManager:e}=await Promise.resolve().then(()=>Z);this.quizRoomManager=new e(this.actualRoomName,this),this.quizRoomManager.addUser(this.myClientId,this.myUsername),this.quizRoomManager.startPeriodicCollection(),console.log("AI quiz room initialized for:",this.actualRoomName)}catch(e){console.error("Failed to initialize AI quiz room:",e),console.log("AI functionality will be disabled for this session"),this.quizRoomManager=null}}handleAIMessage(e,t){if(this.quizRoomManager)switch(e.type){case"collect_content":this.quizRoomManager.handleContentCollectionRequest(e);break;case"content_response":this.quizRoomManager.handleContentResponse(e);break;case"quiz_message":this.onMessage("AI GO!",e,!1);break;case"leader_selected":this.onInfoMessage(`New leader selected: ${e.leaderName}`,"info");break}}updateRoomUser(e,t,o){this.quizRoomManager&&(o==="join"?this.quizRoomManager.addUser(e,t):o==="leave"&&this.quizRoomManager.removeUser(e))}getCurrentPageData(){return{text:document.body.textContent||"No content available",url:window.location.href,title:document.title,timestamp:Date.now()}}}const T=`
/* Widget container styles */
.widget-container {
    --primary-color: #667eea;
    --primary-color-dark: #5a6fd8;
    --background-color: white;
    --secondary-background-color: #f8f9fa;
    --tertiary-background-color: #e9ecef;
    --text-color: #495057;
    --secondary-text-color: #6c757d;
    --heading-text-color: #2c3e50;
    --border-color: #e1e5e9;
    --shadow-color: #e1e5e9;
    --error-color: #721c24;
    --error-background-color: #f8d7da;
    --success-color: #155724;
    --success-background-color: #d4edda;
    --own-message-background-color: #667eea;
    --own-message-text-color: white;
    --presence-label-background-color: #2c3e50;
    --presence-label-text-color: white;
    --presence-label-border-color: #34495e;
    --user-count-background-color: #ff4757;
    --user-count-text-color: white;
    --user-count-border-color: white;
    --focus-ring-color: #e3f2fd;

    position: fixed;
    bottom: 20px;
    left: 20px;
    z-index: 10000;
    font-family: "Helvetica", "Arial", sans-serif;
    /* Remove max-width constraint to allow expansion beyond viewport */
    display: block;
    width: auto;
    height: auto;
    background: transparent;
    border: none;
    margin: 0;
    padding: 0;
}

.widget-container[theme="dark"] {
    --primary-color: #7b8cde;
    --primary-color-dark: #6170c4;
    --background-color: #2c3e50;
    --secondary-background-color: #34495e;
    --tertiary-background-color: #4a6fa5;
    --text-color: #ecf0f1;
    --secondary-text-color: #bdc3c7;
    --heading-text-color: #ffffff;
    --border-color: #4a6fa5;
    --shadow-color: rgba(0, 0, 0, 0.2);
    --error-color: #e74c3c;
    --error-background-color: #f2dede;
    --success-color: #2ecc71;
    --success-background-color: #dff0d8;
    --own-message-background-color: #7b8cde;
    --own-message-text-color: #ffffff;
    --presence-label-background-color: #ecf0f1;
    --presence-label-text-color: #2c3e50;
    --presence-label-border-color: #bdc3c7;
    --user-count-background-color: #e74c3c;
    --user-count-text-color: #ffffff;
    --user-count-border-color: #2c3e50;
    --focus-ring-color: rgba(123, 140, 222, 0.5);
}

.widget-toggle {
    width: 48px;
    height: 48px;
    background: var(--primary-color);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 2px 8px var(--shadow-color);
    transition: all 0.3s ease;
    position: relative;
    border: 2px solid var(--primary-color-dark);
}

.widget-toggle:hover {
    background: var(--primary-color-dark);
    transform: scale(1.05);
    box-shadow: 0 4px 12px #adb5bd;
}

.toggle-icon {
    font-size: 18px;
    color: white;
}

.user-count {
    position: absolute;
    top: -3px;
    right: -3px;
    background: #ff4757;
    color: white;
    border-radius: 50%;
    width: 16px;
    height: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: bold;
    border: 2px solid white;
}

.widget-panel {
    display: flex;
    flex-direction: column;
    position: absolute;
    bottom: 60px;
    left: 0;
    width: 320px;
    height: 440px;
    background: var(--background-color);
    border-radius: 12px;
    box-shadow: 0 4px 20px var(--shadow-color);
    border: 2px solid var(--border-color);
    overflow: hidden;
    animation: slideUp 0.3s ease;
    /* Ensure panel doesn't get cut off */
    max-height: calc(100vh - 100px);
    min-height: 200px;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.widget-header {
    background: var(--secondary-background-color);
    color: #495057;
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
}

.widget-title {
    font-weight: 700;
    font-size: 14px;
    font-family: "Helvetica", "Arial", sans-serif;
}

.widget-controls {
    display: flex;
    gap: 8px;
}

.widget-controls button {
    background: transparent;
    border: 1px solid var(--border-color);
    color: #6c757d;
    width: 24px;
    height: 24px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
}

.widget-controls button:hover {
    background: #e9ecef;
    border-color: #adb5bd;
    color: #495057;
}

.widget-content {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
    background: var(--background-color);
    /* Custom scrollbar styling */
    scrollbar-width: thin;
    scrollbar-color: rgba(102, 126, 234, 0.3) transparent;
}

.resize-handle-ne {
    position: absolute;
    top: 0;
    right: 0;
    width: 12px;
    height: 12px;
    cursor: nesw-resize;
    z-index: 100;
    background-image: linear-gradient(135deg, #ccc 25%, transparent 25%),
                      linear-gradient(225deg, #ccc 25%, transparent 25%);
    background-size: 6px 6px;
    background-repeat: no-repeat;
    background-position: right top;
}

.widget-content::-webkit-scrollbar {
    width: 6px;
}

.widget-content::-webkit-scrollbar-track {
    background: var(--secondary-background-color);
}

.widget-content::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: 3px;
}

.widget-content::-webkit-scrollbar-thumb:hover {
    background: var(--primary-color-dark);
}

.plugin-tab {
    padding: 8px 16px;
    background: var(--secondary-background-color);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    gap: 8px;
}

.plugin-tab button {
    background: none;
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-family: "Helvetica", "Arial", sans-serif;
    font-weight: 700;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 6px;
}

.plugin-tab button.active {
    background: var(--primary-color);
    color: white;
}

.plugin-tab button:hover:not(.active) {
    background: #e9ecef;
}

.plugin-content {
    display: none;
}

.plugin-content.active {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    min-height: 0;
}

/* Chat styles */
.chat-container {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    min-height: 0;
    height: 100%;
    overflow: hidden;
}

.chat-messages {
    flex: 1 1 0;
    overflow-y: auto;
    padding: 12px;
    min-height: 0;
    max-height: calc(100% - 160px);
}

.chat-message {
    margin-bottom: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    max-width: 80%;
    word-wrap: break-word;
    background: var(--background-color);
    border: 1px solid var(--border-color);
}

.chat-message.own {
    background: var(--primary-color);
    color: white;
    margin-left: auto;
    border: 1px solid var(--primary-color-dark);
}

.chat-message.other {
    background: var(--background-color);
    border: 1px solid var(--border-color);
}

.chat-message.system {
    font-style: italic;
    text-align: left;
    color: var(--secondary-text-color);
    background-color: var(--tertiary-background-color);
    max-width: 90%;
    margin: 8px 0;
    padding: 4px 8px;
}

.system-message {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 8px;
    flex-wrap: wrap;
}

.system-icon {
    font-size: 14px;
}

.system-text {
    font-family: "Tinos", serif;
    font-weight: 400;
}

.system-link {
    color: var(--primary-color);
    text-decoration: none;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
    background: rgba(var(--primary-color-rgb), 0.1);
    transition: all 0.2s ease;
    border: 1px solid rgba(var(--primary-color-rgb), 0.2);
    font-size: 12px;
    white-space: nowrap;
}

.system-link:hover {
    background: rgba(var(--primary-color-rgb), 0.2);
    border-color: rgba(var(--primary-color-rgb), 0.4);
    text-decoration: underline;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.system-link:active {
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.chat-message .username {
    font-weight: 700;
    font-size: 11px;
    font-family: "Helvetica", "Arial", sans-serif;
    margin-bottom: 2px;
    opacity: 1;
}

.chat-message .message-text {
    font-family: "Roboto", "Helvetica", "Arial", sans-serif;
    font-weight: 400;
}

.chat-input-container {
    flex: 0 0 auto;
    padding: 12px;
    margin-bottom: 8px;
    height: 144px;
    min-height: 100px;
}

.gemini-input-area {
    background: var(--background-color);
    border: 1px solid var(--border-color);
    border-radius: 24px;
    padding: 12px 12px 16px 12px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    gap: 2px;
    flex-direction: column;
}

.gemini-input-area:focus-within {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
}

.input-top-row {
    display: flex;
    align-items: flex-start;
}

.gemini-textarea {
    flex: 1;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text-color);
    font-size: 14px;
    font-family: inherit;
    line-height: 1.4;
    resize: none;
    min-height: 20px;
    max-height: 120px;
    overflow-y: auto;
    padding: 0 0 4px 0;
    margin: 0;
    width: 100%;
}

.gemini-textarea::placeholder {
    color: var(--secondary-text-color);
}

.input-bottom-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 0;
}

.input-left-actions {
    display: flex;
    align-items: center;
    gap: 4px;
}

.input-right-actions {
    display: flex;
    align-items: center;
}

.send-btn {
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
    flex-shrink: 0;
}

.send-btn:hover {
    background: var(--primary-color-dark);
    transform: scale(1.05);
}

.send-btn:disabled {
    background: var(--secondary-text-color);
    cursor: not-allowed;
    transform: none;
}

.send-btn svg {
    width: 16px;
    height: 16px;
}

.typing-indicator {
    flex: 0 0 auto;
    padding: 8px 12px;
    font-style: italic;
    color: #6c757d;
    font-size: 12px;
}


/* User Stats Styles */
.user-stats {
    background: var(--secondary-background-color);
    border-bottom: 1px solid var(--border-color);
    font-size: 11px;
}

.user-stats-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    cursor: pointer;
    user-select: none;
}

.user-stats-title {
    color: var(--text-color);
    font-size: 11px;
    font-weight: 500;
    font-family: "Roboto", "Helvetica", "Arial", sans-serif;
}

.user-stats-toggle {
    background: none;
    border: none;
    padding: 2px;
    cursor: pointer;
    color: var(--secondary-text-color);
    transition: all 0.2s ease;
    border-radius: 2px;
}

.user-stats-toggle:hover {
    background: var(--tertiary-background-color);
    color: var(--text-color);
}

.user-stats-toggle svg {
    transition: transform 0.2s ease;
}

.user-stats-toggle.expanded svg {
    transform: rotate(0deg);
}

.user-stats-toggle.collapsed svg {
    transform: rotate(-90deg);
}

.user-stats-controls {
    display: flex;
    align-items: center;
    gap: 4px;
}

.info-btn {
    background: none;
    border: none;
    padding: 2px;
    cursor: pointer;
    color: var(--secondary-text-color);
    transition: all 0.2s ease;
    border-radius: 2px;
}

.info-btn:hover {
    background: var(--tertiary-background-color);
    color: var(--text-color);
}

.info-btn svg {
    transition: transform 0.2s ease;
}

.info-btn:hover svg {
    transform: scale(1.1);
}

/* Information Modal Styles */
.info-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 10001;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: opacity 0.2s ease;
}

.info-modal.show {
    opacity: 1;
}

.info-modal.show .info-modal-content {
    transform: scale(1);
}

.info-modal-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(2px);
}

.info-modal-content {
    position: relative;
    background: var(--background-color);
    border-radius: 8px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    border: 1px solid var(--border-color);
    transform: scale(0.9);
    transition: transform 0.2s ease;
}

.info-modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-color);
    background: var(--secondary-background-color);
    border-radius: 8px 8px 0 0;
}

.info-modal-header h3 {
    margin: 0;
    color: var(--heading-text-color);
    font-size: 18px;
    font-weight: 600;
}

.info-modal-close {
    background: none;
    border: none;
    padding: 4px;
    cursor: pointer;
    color: var(--secondary-text-color);
    transition: all 0.2s ease;
    border-radius: 4px;
}

.info-modal-close:hover {
    background: var(--tertiary-background-color);
    color: var(--text-color);
}

.info-modal-body {
    padding: 20px;
}

.feature-section {
    margin-bottom: 20px;
}

.feature-section:last-child {
    margin-bottom: 0;
}

.feature-section h4 {
    margin: 0 0 8px 0;
    color: var(--heading-text-color);
    font-size: 14px;
    font-weight: 600;
}

.feature-section p {
    margin: 0;
    color: var(--text-color);
    font-size: 13px;
    line-height: 1.4;
}

.user-stats-content {
    display: none !important;
    justify-content: space-between;
    flex-wrap: nowrap;
    gap: 4px;
    padding: 6px 8px;
    transition: all 0.3s ease;
    overflow: hidden;
}

.user-stats-content.expanded {
    display: flex !important;
}

.user-stats-content.collapsed {
    display: none !important;
}

.stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 50px;
    flex: 1;
}

.stat-item.room-status {
    padding: 2px 4px;
    border-radius: 3px;
}

.stat-item.room-status.connected {
    background: #d4edda;
    color: #155724;
}

.stat-item.room-status.disconnected {
    background: #f8d7da;
    color: #721c24;
}

.stat-label {
    color: var(--secondary-text-color);
    font-size: 9px;
    margin-bottom: 1px;
    font-family: "Roboto", "Helvetica", "Arial", sans-serif;
}

.stat-item.room-status .stat-label {
    color: inherit;
    font-weight: 500;
}

.stat-value {
    color: var(--primary-color);
    font-weight: bold;
    font-size: 11px;
    font-family: "Roboto", "Helvetica", "Arial", sans-serif;
}

.stat-item.room-status .stat-value {
    color: inherit;
    font-weight: 600;
}

/* Chat Actions Styles */
.chat-actions {
    display: flex;
    justify-content: right;
    gap: 8px;
}

.action-btn {
    background: none;
    border: none;
    padding: 6px;
    cursor: pointer;
    color: var(--secondary-text-color);
    border-radius: 50%;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
}

.action-btn:hover {
    background: var(--tertiary-background-color);
    color: var(--text-color);
}

.action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.action-btn:disabled:hover {
    background: none;
    color: var(--secondary-text-color);
}

.action-btn svg {
    width: 14px;
    height: 14px;
}

/* Room Selection Styles */
.room-selection {
    padding: 20px;
}

.room-header {
    text-align: center;
    margin-bottom: 20px;
}

.room-header h3 {
    margin: 0 0 8px 0;
    color: #2c3e50;
    font-size: 18px;
    font-family: "Helvetica", "Arial", sans-serif;
    font-weight: 700;
}

.room-header p {
    margin: 0;
    color: #6c757d;
    font-size: 14px;
    font-family: "Roboto", "Helvetica", "Arial", sans-serif;
    font-weight: 400;
}

.room-tabs {
    display: flex;
    margin-bottom: 20px;
    background: var(--secondary-background-color);
    border-radius: 8px;
    padding: 4px;
}

.room-tab {
    flex: 1;
    padding: 8px 16px;
    border: none;
    background: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 700;
    font-family: "Helvetica", "Arial", sans-serif;
    transition: all 0.2s ease;
    color: #6c757d;
}

.room-tab.active {
    background: var(--background-color);
    color: var(--primary-color);
    box-shadow: 0 2px 4px var(--shadow-color);
}

.room-tab:hover:not(.active) {
    color: #495057;
}

.room-form {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.form-group label {
    font-size: 14px;
    font-weight: 700;
    font-family: "Helvetica", "Arial", sans-serif;
    color: #495057;
}

.form-group input {
    padding: 10px 12px;
    border: 1px solid #e1e5e9;
    border-radius: 8px;
    font-size: 14px;
    font-family: "Roboto", "Helvetica", "Arial", sans-serif;
    font-weight: 400;
    background: var(--background-color);
    transition: border-color 0.2s ease;
}

.form-group input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px #e3f2fd;
}

.connect-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px 20px;
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    font-family: "Helvetica", "Arial", sans-serif;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-top: 8px;
}

.connect-btn:hover {
    background: var(--primary-color-dark);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px var(--primary-color);
}

.connect-btn:active {
    transform: translateY(0);
}

/* Responsive adjustments */
@media (max-width: 480px) {
    .widget-panel {
        width: calc(100vw - 40px);
        max-width: 320px;
        left: 0;
        right: 0;
        margin: 0 auto;
    }
}

@media (max-height: 600px) {
    .widget-panel {
        max-height: calc(100vh - 100px);
    }
};
`,z=`
<div class="widget-container">
    <div class="widget-toggle" id="widget-toggle">
        <div class="toggle-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
        </div>
        <div class="user-count" id="user-count">0</div>
    </div>
    <div class="widget-panel" id="widget-panel" style="display: none;">
        <div class="resize-handle-ne"></div>
        <div class="widget-header">
            <div class="widget-title">Collaborative</div>
            <div class="widget-controls">
                <button class="gamification-btn" id="gamification-btn" title="Gamification">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
                    </svg>
                    <span class="gamification-play-text">Play</span>
                </button>
                <button class="minimize-btn" id="minimize-btn" title="Minimize">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                </button>
                <button class="exit-btn" id="exit-btn" title="Exit Room">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                        <polyline points="16 17 21 12 16 7"></polyline>
                        <line x1="21" y1="12" x2="9" y2="12"></line>
                    </svg>
                </button>
                
            </div>
        </div>
        <div class="widget-content" id="widget-content">
            <!-- Plugin content will be inserted here -->
        </div>
    </div>
</div>
`;class R{constructor(e,t){this.widgetPanel=e,this.resizeHandle=t,this.isResizing=!1,this.originalWidth=0,this.originalHeight=0,this.originalX=0,this.originalY=0,this.originalMouseX=0,this.originalMouseY=0,this.initResize=this.initResize.bind(this),this.resize=this.resize.bind(this),this.stopResize=this.stopResize.bind(this),console.log("WidgetResizer constructor:",{widgetPanel:this.widgetPanel,resizeHandle:this.resizeHandle,resizeHandleExists:!!this.resizeHandle}),this.resizeHandle?(this.resizeHandle.addEventListener("mousedown",this.initResize),console.log("Resize handle event listener added")):console.error("Resize handle not found!"),this.loadSize()}initResize(e){e.preventDefault(),this.isResizing=!0;const t=this.widgetPanel.getBoundingClientRect(),o=this.widgetPanel.parentElement.getBoundingClientRect();this.originalWidth=t.width,this.originalHeight=t.height,this.originalX=t.left-o.left,this.originalY=t.top-o.top,this.originalMouseX=e.pageX,this.originalMouseY=e.pageY,console.log("Init resize:",{originalWidth:this.originalWidth,originalHeight:this.originalHeight,originalX:this.originalX,originalY:this.originalY,mouseX:this.originalMouseX,mouseY:this.originalMouseY,rect:t,parentRect:o}),document.addEventListener("mousemove",this.resize),document.addEventListener("mouseup",this.stopResize)}resize(e){if(!this.isResizing)return;const t=e.pageX-this.originalMouseX,o=e.pageY-this.originalMouseY,s=this.originalWidth+t,n=this.originalHeight+o,i=200,a=200,r=Math.max(s,i),c=Math.max(n,a);console.log("Resize:",{dx:t,dy:o,originalWidth:this.originalWidth,originalHeight:this.originalHeight,newWidth:s,newHeight:n,finalWidth:r,finalHeight:c,originalX:this.originalX,originalY:this.originalY}),this.widgetPanel.style.width=r+"px",this.widgetPanel.style.height=c+"px",this.widgetPanel.style.left=this.originalX+"px",this.widgetPanel.style.top=this.originalY+"px",this.widgetPanel.style.bottom="auto",this.widgetPanel.style.right="auto"}stopResize(){this.isResizing=!1,document.removeEventListener("mousemove",this.resize),document.removeEventListener("mouseup",this.stopResize),this.saveSize()}saveSize(){const e={width:this.widgetPanel.style.width,height:this.widgetPanel.style.height,left:this.widgetPanel.style.left,top:this.widgetPanel.style.top};localStorage.setItem("widgetSize",JSON.stringify(e))}loadSize(){const e=JSON.parse(localStorage.getItem("widgetSize"));e&&(this.widgetPanel.style.width=e.width,this.widgetPanel.style.height=e.height,this.widgetPanel.style.left=e.left,this.widgetPanel.style.top=e.top,this.widgetPanel.style.right="auto")}}const M=`
.gamification-btn {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    padding: 6px 12px !important;
    border: 1px solid #ddd !important;
    background: #f8f9fa !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    width: auto !important;
    height: auto !important;
    min-width: 60px !important;
}

.gamification-btn:hover {
    background: #e9ecef !important;
    border-color: #ccc !important;
}

.gamification-btn:disabled {
    opacity: 0.6 !important;
    cursor: not-allowed !important;
    background: #f8f9fa !important;
    border-color: #ddd !important;
}

.gamification-btn svg {
    stroke: #FFD700 !important; /* Gold */
    transition: all 0.3s ease;
    width: 16px !important;
    height: 16px !important;
    flex-shrink: 0 !important;
}

.gamification-btn:hover svg {
    filter: drop-shadow(0 0 5px #FFD700) !important;
}

.gamification-btn:disabled svg {
    stroke: #999 !important; /* Gray when disabled */
}

.gamification-btn:disabled:hover svg {
    filter: none !important;
}

.gamification-play-text {
    color: #333 !important; /* Black text */
    font-size: 11px !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}

.gamification-btn:disabled .gamification-play-text {
    color: #999 !important; /* Gray when disabled */
}


.gamification-btn.glowing-border::before {
    content: '';
    position: absolute;
    top: -2px; left: -2px;
    right: -2px; bottom: -2px;
    border: 2px solid transparent;
    border-radius: 10px;
    animation: glow 1.5s infinite;
}

@keyframes glow {
    0% {
        border-color: #FFD700;
        box-shadow: 0 0 5px #FFD700;
    }
    50% {
        border-color: #FFD700;
        box-shadow: 0 0 20px #FFD700;
    }
    100% {
        border-color: #FFD700;
        box-shadow: 0 0 5px #FFD700;
    }
}
`,U=`
.gamification-panel {
    position: fixed; /* Changed to fixed for viewport positioning */
    top: 20px;
    left: 20px;
    width: 400px;
    background-color: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    z-index: 10001;
    display: none; /* Initially hidden */
    color: #333;
    font-family: "Tinos", serif;
}

.gamification-header {
    padding: 8px 12px;
    border-bottom: 1px solid #ddd;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: move; /* Add move cursor to header */
}

.gamification-tabs {
    display: flex;
    border-bottom: 1px solid #ddd;
}

.tab-button {
    padding: 8px 12px;
    cursor: pointer;
    background-color: #f8f9fa;
    border: none;
    border-right: 1px solid #ddd;
    font-size: 13px;
    font-weight: 500;
    color: #666;
}

.tab-button.active {
    background-color: #fff;
    font-weight: bold;
    color: #007bff;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.team-members-list {
    list-style: none;
    padding-left: 15px;
    margin: 0;
    font-size: 11px;
}

.team-members-list li {
    padding: 2px 0;
}

.gamification-header .drag-handle {
    display: inline-block;
    width: 20px;
    height: 20px;
    color: #aaa;
}

.gamification-table {
    width: 100%;
    border-collapse: collapse;
}

.gamification-table th,
.gamification-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
    font-size: 12px;
}

.gamification-table th {
    font-weight: bold;
    background-color: #f8f9fa;
}

.gamification-table a {
    color: #007bff;
    text-decoration: none;
}

.gamification-table a:hover {
    text-decoration: underline;
}

.powerup-icon, .star-icon {
    display: inline-block;
    width: 16px;
    height: 16px;
}

.powerup-icon svg {
    stroke: #4A90E2; /* Blue */
}

.star-icon svg {
    stroke: #F5A623; /* Orange */
    fill: #F5A623;
}

.star-display {
    display: flex;
    align-items: center;
    gap: 4px;
}

.total-score {
    font-size: 11px;
    color: #666;
    font-style: italic;
    margin-left: 4px;
}

.badge {
    position: relative;
    display: inline-block;
    transition: transform 0.2s ease;
}

.badge:hover {
    transform: scale(1.2);
    z-index: 10;
}

.badge .tooltip {
    visibility: hidden;
    width: 120px;
    background-color: #333;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 0;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -60px;
    opacity: 0;
    transition: opacity 0.3s;
}

.badge:hover .tooltip {
    visibility: visible;
    opacity: 1;
}

.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    z-index: 10002;
    display: none; /* Hidden by default */
}

.modal-content {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #fff;
    padding: 20px;
    border-radius: 8px;
    width: 90%;
    max-width: 500px;
    z-index: 10003;
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
    margin-bottom: 10px;
}

.modal-close-btn {
    border: none;
    background: transparent;
    font-size: 24px;
    cursor: pointer;
}

.games-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.games-list li {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    background: #f8f9fa;
    border-radius: 4px;
    padding: 5px;
    transition: background 0.2s ease;
}

.games-list li:hover {
    background: #e9ecef;
}

.game-thumbnail {
    width: 60px;
    height: 60px;
    border-radius: 4px;
    margin-right: 15px;
    object-fit: cover;
}

.game-info {
    flex-grow: 1;
}

.game-title {
    font-weight: bold;
    color: #007bff;
    text-decoration: none;
}

.game-description {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}

.modal-game-image {
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: 8px;
    margin-bottom: 15px;
}

.modal-footer {
    display: flex;
    justify-content: center;
    margin-top: 20px;
    padding-top: 15px;
    border-top: 1px solid #eee;
}

.configure-game-btn {
    background: #007bff;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s ease;
}

.configure-game-btn:hover {
    background: #0056b3;
}

.team-assignment-modal {
    max-width: 600px;
}

.team-assignment-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}

.team-assignment-table th,
.team-assignment-table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

.team-assignment-table th {
    background-color: #f8f9fa;
    font-weight: bold;
}

.team-select {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    font-size: 14px;
}

.start-game-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 15px 30px;
    border-radius: 25px;
    font-size: 18px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.start-game-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.6s;
}

.start-game-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5);
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}

.start-game-btn:hover::before {
    left: 100%;
}

.start-game-btn:active {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.countdown-overlay {
    background: rgba(0,0,0,0.9);
}

.countdown-content {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #1a1a1a;
    border: 2px solid #333;
    border-radius: 12px;
    padding: 40px;
    z-index: 10004;
    text-align: center;
}

.countdown-header {
    margin-bottom: 30px;
}

.countdown-title {
    color: #00ff00;
    font-size: clamp(1.2rem, 3vw, 1.8rem);
    font-weight: bold;
    margin: 0 0 10px 0;
    text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00;
    animation: glow-pulse 2s ease-in-out infinite alternate;
}

.countdown-subtitle {
    color: #00ff00;
    font-size: clamp(1rem, 2.5vw, 1.4rem);
    margin: 0;
    opacity: 0.8;
    text-shadow: 0 0 5px #00ff00;
}

@keyframes glow-pulse {
    0% {
        text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00;
    }
    100% {
        text-shadow: 0 0 15px #00ff00, 0 0 25px #00ff00, 0 0 35px #00ff00;
    }
}

.ascii-display {
    color: #00ff00;
    font-family: 'Courier New', monospace;
    font-size: clamp(1rem, 4vw, 2rem);
    line-height: 1.1;
    text-align: center;
    margin: 0;
    white-space: pre;
}

.configure-header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid #eee;
}

.back-button {
    background: #6c757d;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.2s ease;
}

.back-button:hover {
    background: #5a6268;
}

.configure-body {
    margin-bottom: 20px;
}

.configure-body p {
    margin-bottom: 15px;
    color: #666;
    font-size: 14px;
}

.configure-footer {
    display: flex;
    justify-content: center;
    padding-top: 15px;
    border-top: 1px solid #eee;
}

.team-configuration-modal {
    max-width: 700px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.team-naming-section {
    margin-bottom: 25px;
    padding-bottom: 20px;
    border-bottom: 1px solid #eee;
}

.team-naming-section h4 {
    margin-bottom: 15px;
    color: #333;
    font-size: 16px;
}

.team-names-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
}

.team-name-input {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.team-name-input label {
    font-size: 14px;
    font-weight: 500;
    color: #555;
}

.team-name-field {
    padding: 8px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    background: white;
}

.team-name-field:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

.player-assignment-section h4 {
    margin-bottom: 10px;
    color: #333;
    font-size: 16px;
}

.player-assignment-section p {
    margin-bottom: 15px;
    color: #666;
    font-size: 14px;
}

.drag-instructions {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 15px;
    margin-bottom: 20px;
}

.drag-instructions h4 {
    margin-bottom: 10px;
    color: #333;
    font-size: 16px;
    font-weight: 600;
}

.instruction-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 5px 0;
    color: #666;
    font-size: 14px;
}

.instruction-icon {
    color: #007bff;
    font-weight: bold;
    width: 16px;
    text-align: center;
}

.teams-container {
    margin-bottom: 25px;
}

.teams-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
    margin-bottom: 15px;
}

.team-section {
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 15px;
    background: #fff;
    min-height: 120px;
}

.team-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
}

.team-name-field {
    flex: 1;
    padding: 8px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 500;
    background: #f8f9fa;
}

.team-name-field:focus {
    outline: none;
    border-color: #007bff;
    background: white;
}

.remove-team-btn {
    background: #dc3545;
    color: white;
    border: none;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
}

.remove-team-btn:hover {
    background: #c82333;
}

.team-players {
    min-height: 60px;
    border: 2px dashed #ddd;
    border-radius: 4px;
    padding: 10px;
    transition: all 0.2s ease;
}

.team-players.drag-over {
    border-color: #007bff;
    background-color: rgba(0, 123, 255, 0.1);
}

.player-item {
    background: #007bff;
    color: white;
    padding: 6px 12px;
    border-radius: 20px;
    margin: 4px 0;
    cursor: grab;
    font-size: 14px;
    transition: all 0.2s ease;
    user-select: none;
}

.player-item:hover {
    background: #0056b3;
    transform: translateY(-1px);
}

.player-item.dragging {
    opacity: 0.5;
    transform: rotate(5deg);
}

.add-team-btn {
    background: #28a745;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.2s ease;
}

.add-team-btn:hover {
    background: #218838;
}

.unassigned-players-section {
    margin-top: 20px;
}

.unassigned-players-section h4 {
    margin-bottom: 10px;
    color: #333;
    font-size: 16px;
}

.unassigned-players {
    min-height: 80px;
    border: 2px dashed #ddd;
    border-radius: 6px;
    padding: 15px;
    background: #f8f9fa;
    transition: all 0.2s ease;
}

.unassigned-players.drag-over {
    border-color: #6c757d;
    background-color: rgba(108, 117, 125, 0.1);
}

.unassigned-players .player-item {
    display: inline-block;
    margin: 4px 8px 4px 0;
}
`;class P{constructor(){this.teamNames=["Team Alpha","Team Bravo","Team Charlie","Team Delta"]}getTeams(e){const t={};this.teamNames.forEach(o=>{t[o]={name:o,members:[],score:0,badges:[],powerups:[]}}),e.forEach((o,s)=>{const n=this.teamNames[s%this.teamNames.length];t[n].members.push(o)});for(const o in t){const s=t[o];s.score=s.members.reduce((n,i)=>n+(i.stars||0),0),s.members.length>0&&s.badges.push({icon:"🔥",text:"On Fire"})}return Object.values(t).filter(o=>o.members.length>0)}}class I{constructor(e=3e4){this.timeLimit=e,this.startTime=null,this.remainingTime=e,this.isRunning=!1,this.isPaused=!1,this.timerInterval=null,this.callbacks={onTick:null,onComplete:null,onPause:null,onResume:null}}startTimer(){if(this.isRunning&&!this.isPaused){console.warn("Timer is already running");return}this.startTime=Date.now(),this.isRunning=!0,this.isPaused=!1,this.timerInterval=setInterval(()=>{this.tick()},100),console.log("🎮 Kapoot timer started:",this.timeLimit+"ms")}pauseTimer(){!this.isRunning||this.isPaused||(clearInterval(this.timerInterval),this.isPaused=!0,this.callbacks.onPause&&this.callbacks.onPause(this.remainingTime),console.log("🎮 Kapoot timer paused at:",this.remainingTime+"ms"))}resumeTimer(){!this.isRunning||!this.isPaused||(this.timerInterval=setInterval(()=>{this.tick()},100),this.isPaused=!1,this.callbacks.onResume&&this.callbacks.onResume(this.remainingTime),console.log("🎮 Kapoot timer resumed at:",this.remainingTime+"ms"))}stopTimer(){clearInterval(this.timerInterval),this.isRunning=!1,this.isPaused=!1,this.timerInterval=null,console.log("🎮 Kapoot timer stopped")}resetTimer(){this.stopTimer(),this.remainingTime=this.timeLimit,this.startTime=null}tick(){if(!this.isRunning||this.isPaused)return;const e=Date.now()-this.startTime;this.remainingTime=Math.max(0,this.timeLimit-e),this.callbacks.onTick&&this.callbacks.onTick(this.remainingTime),this.remainingTime<=0&&(this.stopTimer(),this.callbacks.onComplete&&this.callbacks.onComplete())}calculateScore(e,t,o){if(!o)return 0;const s=this.timeLimit-e,n=Math.max(0,s/this.timeLimit),i=Math.floor(t*n);return console.log("🎮 Score calculation:",{responseTime:e,basePoints:t,timeRemaining:s,scoreMultiplier:n,finalScore:i}),i}getTimeBonus(e){return Math.max(0,e/this.timeLimit)}setCallbacks(e){this.callbacks={...this.callbacks,...e}}getState(){return{isRunning:this.isRunning,isPaused:this.isPaused,remainingTime:this.remainingTime,timeLimit:this.timeLimit,elapsed:this.timeLimit-this.remainingTime}}static formatTime(e){return`${Math.ceil(e/1e3)}s`}static getUrgencyLevel(e){const t=e/3e4;return t>.5?"safe":t>.2?"warning":"critical"}}class x{static AI_ENDPOINT="https://proxy-worker.mlsysbook.workers.dev/ai";static AI_STREAM_ENDPOINT="https://proxy-worker.mlsysbook.workers.dev/ai/stream";static DEFAULT_MODEL="openai/gpt-oss-20b";static DEFAULT_TEMPERATURE=.7;static MAX_TOKENS=1e3;static PROVIDERS=[{name:"groq",model:"openai/gpt-oss-20b",url:"https://api.groq.com/openai/v1/chat/completions"},{name:"gemini",model:"gemini-2.5-flash",url:"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"},{name:"openai",model:"deepseek/deepseek-chat-v3.1:free",url:"https://openrouter.ai/api/v1/chat/completions"},{name:"mistral",model:"mistral-tiny",url:"https://api.mistral.ai/v1/chat/completions"}];static async generateQuiz(e){try{const t=this.buildQuizPrompt(e),o=await this.callAIProxy(t),s=this.parseQuizResponse(o);return s.quizId=`quiz-${Date.now()}-${Math.random().toString(36).substr(2,9)}`,s.metadata={generatedAt:new Date().toISOString(),userCount:e.userCount,contentSources:e.content.map(n=>n.pageTitle),totalCharacters:e.totalCharacters},s}catch(t){return console.error("Quiz generation failed:",t),this.createFallbackQuiz(e)}}static buildQuizPrompt(e){const t=e.content.map(o=>`${o.username} (${o.pageTitle}): ${o.sampledText}`).join(`

`);return console.log("🎯 QuizGenerator - Content being sent to AI:",{userCount:e.userCount,totalCharacters:e.totalCharacters,contentText:t.substring(0,200)+"..."}),{messages:[{role:"system",content:`You are an expert quiz generator. Create a multiple choice quiz based on the collaborative content from multiple users viewing different web pages.

The quiz should:
- Test understanding of the combined topics
- Have 4 answer choices (A, B, C, D)
- Include explanations for each choice
- Be educational and engaging
- Cover the main concepts from the content

Return ONLY a JSON object with this exact structure:
{
  "question": "The main question text here",
  "choices": {
    "A": "First choice text",
    "B": "Second choice text", 
    "C": "Third choice text",
    "D": "Fourth choice text"
  },
  "correctAnswer": "A",
  "explanations": {
    "A": "Explanation for why A is correct",
    "B": "Explanation for why B is incorrect", 
    "C": "Explanation for why C is incorrect",
    "D": "Explanation for why D is incorrect"
  },
  "difficulty": "beginner|intermediate|advanced",
  "topics": ["topic1", "topic2", "topic3"]
}`},{role:"user",content:`Generate a quiz based on this collaborative content from ${e.userCount} users:

${t}`}],model:this.DEFAULT_MODEL,temperature:this.DEFAULT_TEMPERATURE,max_tokens:this.MAX_TOKENS}}static async callAIProxy(e){let t=null;for(const o of this.PROVIDERS)try{console.log(`🎯 Trying AI provider: ${o.name} (${o.model})`);const s=await fetch(`${this.AI_ENDPOINT}?url=${encodeURIComponent(o.url)}&provider=${o.name}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...e,model:o.model,temperature:this.DEFAULT_TEMPERATURE,max_tokens:this.MAX_TOKENS})});if(!s.ok){const i=await s.text();console.warn(`❌ ${o.name} failed: ${s.status} ${s.statusText}`,i),t=new Error(`${o.name} failed: ${s.status} ${s.statusText}`);continue}const n=await s.json();return console.log(`✅ ${o.name} succeeded!`),this.parseSingleResponse(n)}catch(s){console.warn(`❌ ${o.name} error:`,s.message),t=s;continue}throw new Error(`All AI providers failed. Last error: ${t?.message||"Unknown error"}`)}static parseSingleResponse(e){return e.choices?.[0]?.message?.content?e.choices[0].message.content:e.text?e.text:e.message?e.message:(console.warn("Unexpected response format:",e),JSON.stringify(e))}static async parseStreamingResponse(e){const t=e.body?.getReader();if(!t)throw new Error("No response body");let o="";const s=new TextDecoder;try{for(;;){const{done:n,value:i}=await t.read();if(n)break;const r=s.decode(i,{stream:!0}).split(`
`);for(const c of r)if(c.startsWith("data: ")){const l=c.slice(6).trim();if(l==="[DONE]")break;try{const d=JSON.parse(l);d.choices?.[0]?.delta?.content&&(o+=d.choices[0].delta.content)}catch{}}}}finally{t.releaseLock()}return o}static parseQuizResponse(e){try{const t=e.match(/\{[\s\S]*\}/);if(!t)throw new Error("No JSON found in response");const o=JSON.parse(t[0]);return this.validateQuizData(o),o}catch(t){throw console.error("Failed to parse quiz response:",t),new Error("Invalid quiz response format")}}static validateQuizData(e){const t=["question","choices","correctAnswer","explanations","difficulty","topics"];for(const o of t)if(!e[o])throw new Error(`Missing required field: ${o}`);if(!e.choices.A||!e.choices.B||!e.choices.C||!e.choices.D)throw new Error("Missing required choices A, B, C, D");if(!["A","B","C","D"].includes(e.correctAnswer))throw new Error("Invalid correct answer")}static createFallbackQuiz(e){return{quizId:`quiz-${Date.now()}-${Math.random().toString(36).substr(2,9)}`,question:"What is the main topic covered in the collaborative content?",choices:{A:"Technology and Innovation",B:"Science and Research",C:"Business and Economics",D:"Education and Learning"},correctAnswer:"A",explanations:{A:"Based on the collaborative content, the main focus appears to be on technology and innovation topics.",B:"While science and research may be relevant, the content seems more focused on technology.",C:"Business and economics topics don't appear to be the primary focus of the content.",D:"Education and learning are important but not the main topic of the collaborative content."},difficulty:"intermediate",topics:["collaboration","content analysis"],metadata:{generatedAt:new Date().toISOString(),userCount:e.userCount,contentSources:e.content.map(t=>t.pageTitle),totalCharacters:e.totalCharacters,isFallback:!0}}}}class D{constructor(e){this.roomManager=e,this.currentQuiz=null,this.quizHistory=[]}async generateFromRoomContent(e,t=null){try{console.log("🎮 Generating Kapoot quiz from room content...");let o;if(t&&t.content&&t.content.length>0?(console.log("🎮 Using pre-collected content:",t),o=t.content):(console.log("🎮 Collecting new content..."),o=await this.collectEnhancedContent()),!o||o.length===0)throw new Error("No content available for quiz generation");const s=await this.generateQuizWithAI(o,e),n=this.validateQuiz(s),i=this.formatQuizForDisplay(n,o);return this.currentQuiz=i,this.quizHistory.push(i),console.log("🎮 Quiz generated successfully:",i.quizId),i}catch(o){return console.error("🎮 Failed to generate quiz:",o),this.createFallbackQuiz(e)}}async collectEnhancedContent(){try{const e=await this.roomManager.collectContentFromAll();if(console.log("🎮 Raw user responses:",e.length,"responses"),e.length===0)return console.log("🎮 No content collected, creating basic content for testing"),[{userId:"current-user",username:"Current User",content:{text:document.body.textContent||"Sample content for quiz generation",url:window.location.href,title:document.title||"Current Page",timestamp:Date.now()},sourceUrl:window.location.href,sourceTitle:document.title||"Current Page"}];const t=e.map(o=>({userId:o.userId,username:o.username,content:{text:o.content.text,url:o.content.url||window.location.href,title:o.content.title||document.title,timestamp:o.content.timestamp||Date.now()},sourceUrl:o.content.url||window.location.href,sourceTitle:o.content.title||document.title}));return console.log("🎮 Enhanced content collected:",t.length,"sources"),t}catch(e){return console.error("🎮 Failed to collect content:",e),[{userId:"fallback-user",username:"Fallback User",content:{text:document.body.textContent||"Fallback content for quiz generation",url:window.location.href,title:document.title||"Fallback Page",timestamp:Date.now()},sourceUrl:window.location.href,sourceTitle:document.title||"Fallback Page"}]}}async generateQuizWithAI(e,t){t.map(n=>n.name).join(", "),e.map(n=>`- ${n.username}: "${n.sourceTitle}" (${n.sourceUrl})`).join(`
`);const o={content:e.map(n=>{console.log("🎮 Debugging content structure for user:",n.username,n);let i="";return n.content?.text?i=n.content.text:n.content?.sampledText?i=n.content.sampledText:n.sampledText?i=n.sampledText:typeof n.content=="string"?i=n.content:(console.warn("🎮 Unknown content structure for user:",n.username,n),i="No content available"),{username:n.username,pageTitle:n.sourceTitle||n.pageTitle||n.content?.title||"Unknown Page",sampledText:i.substring(0,500)||"No content available"}}),userCount:e.length,totalCharacters:e.reduce((n,i)=>{const a=i.content?.text?.length||i.content?.sampledText?.length||i.sampledText?.length||0;return n+a},0)};console.log("🎮 Formatted content data for AI:",{userCount:o.userCount,totalCharacters:o.totalCharacters,content:o.content.map(n=>({username:n.username,pageTitle:n.pageTitle,textPreview:n.sampledText.substring(0,100)+"..."}))});const s=x.buildQuizPrompt(o);try{const n=await x.callAIProxy(s),i=JSON.parse(n);return i.quizId=this.generateQuizId(),i.teamQuiz=!0,i.timeLimit=3e4,i.basePoints=100,i.sourceUrls=e.map(a=>({url:a.sourceUrl,title:a.sourceTitle,username:a.username})),i}catch(n){throw console.error("🎮 AI quiz generation failed:",n),n}}validateQuiz(e){const t=["question","choices","correctAnswer","explanations"];for(const n of t)if(!e[n])throw new Error(`Missing required field: ${n}`);const o=e.choices,s=["A","B","C","D"];for(const n of s)if(!o[n]||typeof o[n]!="string")throw new Error(`Invalid choice ${n}`);if(!s.includes(e.correctAnswer))throw new Error("Invalid correct answer");for(const n of s)if(!e.explanations[n])throw new Error(`Missing explanation for choice ${n}`);return console.log("🎮 Quiz validation passed"),e}formatQuizForDisplay(e,t){return{...e,teamResponses:{},metadata:{generatedAt:new Date().toISOString(),userCount:t.length,contentSources:t.map(o=>({url:o.sourceUrl,title:o.sourceTitle,username:o.username})),totalCharacters:t.reduce((o,s)=>o+(s.content?.text?.length||0),0),roomName:this.roomManager?.roomName||"unknown",teamCount:0}}}createMCQuizMessage(e,t=null){e.sourceUrls.map((n,i)=>`${i+1}. [${n.title}](${n.url})${n.username&&n.username!=="System"?` - ${n.username}`:""}`).join(`
`);const o={quizId:e.quizId,question:e.question,choices:e.choices,correctAnswer:e.correctAnswer,explanations:e.explanations,difficulty:e.difficulty,timeLimit:e.timeLimit,basePoints:e.basePoints,sourceUrls:e.sourceUrls,teamResponses:e.teamResponses,quizNumber:e.questionNumber||1,totalQuizzes:e.totalQuestions||5,remainingTime:e.timeLimit,timerExpired:!1,explanationsRevealed:!1,specialFeatures:e.specialFeatures||{},currentUserTeam:t,startTime:Date.now()};return`🎮 **Kapoot Quiz Challenge!**

<quiz-kapoot>
${JSON.stringify(o).replace(/[\r\n\t]/g,"")}
</quiz-kapoot>

🏆 **Scoring:** Points = Base Points × (Time Remaining / Total Time)
⚡ **Strategy:** Quick + Correct = High Score!`}createFallbackQuiz(e){return console.log("🎮 Creating fallback quiz"),{quizId:this.generateQuizId(),question:"What is the main purpose of collaborative learning?",choices:{A:"To compete against other teams",B:"To learn together and share knowledge",C:"To finish tasks quickly",D:"To avoid individual responsibility"},correctAnswer:"B",explanations:{A:"Collaboration isn't just about competition",B:"Correct! Collaborative learning focuses on shared knowledge and mutual support",C:"Speed isn't the primary goal of collaboration",D:"Collaboration actually increases individual responsibility"},difficulty:"beginner",topics:["collaboration","learning"],teamQuiz:!0,timeLimit:3e4,basePoints:100,sourceUrls:[{url:window.location.href,title:document.title,username:"System"}],teamResponses:{},metadata:{generatedAt:new Date().toISOString(),userCount:1,contentSources:[{url:window.location.href,title:document.title,username:"System"}],totalCharacters:0,roomName:"fallback",teamCount:e.length,isFallback:!0}}}generateQuizId(){return`kapoot-quiz-${Date.now()}-${Math.random().toString(36).substr(2,9)}`}getCurrentQuiz(){return this.currentQuiz}getQuizHistory(){return this.quizHistory}clearCurrentQuiz(){this.currentQuiz=null}}const N=`

/* Kapoot Modal Styles */
.kapoot-modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10001;
    backdrop-filter: blur(5px);
}

.kapoot-modal-content {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    width: 90%;
    max-width: 800px;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    border: 2px solid rgba(255, 255, 255, 0.1);
}

.kapoot-modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 30px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px 20px 0 0;
}

.kapoot-modal-title {
    color: #333;
    margin: 0;
    font-size: 24px;
    font-weight: bold;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.kapoot-modal-close {
    background: none;
    border: none;
    color: #333;
    font-size: 28px;
    cursor: pointer;
    padding: 5px;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.2s;
}

.kapoot-modal-close:hover {
    background: rgba(255, 255, 255, 0.1);
}

.kapoot-modal-body {
    padding: 30px;
    color: #333;
}

/* Status Section */
.kapoot-status-section {
    text-align: center;
    padding: 40px 20px;
}

.kapoot-status-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
}

.kapoot-status-icon {
    font-size: 60px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

/* Quiz Section */
.kapoot-quiz-section {
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.kapoot-quiz-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.kapoot-timer-display {
    display: flex;
    align-items: center;
    gap: 15px;
}

.kapoot-timer-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4CAF50, #45a049);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: bold;
    color: #333;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

.kapoot-timer-circle.warning {
    background: linear-gradient(135deg, #FF9800, #F57C00);
}

.kapoot-timer-circle.critical {
    background: linear-gradient(135deg, #F44336, #D32F2F);
    animation: pulse 1s infinite;
}

.kapoot-timer-progress {
    width: 200px;
    height: 8px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    overflow: hidden;
}

.kapoot-timer-bar {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #45a049);
    transition: width 0.1s linear;
}

.kapoot-question-info {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 5px;
}

.kapoot-question-number {
    font-size: 16px;
    font-weight: bold;
    color: #333;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

.kapoot-difficulty {
    font-size: 14px;
    color: #666;
    background: rgba(255, 255, 255, 0.1);
    padding: 4px 12px;
    border-radius: 20px;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

.kapoot-question-container {
    background: rgba(255, 255, 255, 0.1);
    padding: 30px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.kapoot-question {
    font-size: 22px;
    line-height: 1.4;
    margin: 0;
    text-align: center;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

.kapoot-choices-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
}

.kapoot-choice {
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 20px;
    color: #333;
    cursor: pointer;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    gap: 15px;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

.kapoot-choice:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.4);
    transform: translateY(-2px);
}

.kapoot-choice.selected {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    border-color: #4CAF50;
}

.kapoot-choice.disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.kapoot-choice-letter {
    background: rgba(255, 255, 255, 0.2);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 18px;
}

.kapoot-choice-text {
    flex: 1;
    font-size: 16px;
    line-height: 1.3;
    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
}

/* Source Links */
.kapoot-source-links {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.kapoot-source-links h4 {
    margin: 0 0 15px 0;
    color: #333;
}

.kapoot-source-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.kapoot-source-link {
    color: #555;
    text-decoration: none;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.05);
    transition: all 0.2s ease;
}

.kapoot-source-link:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #333;
}

/* Team Scores */
.kapoot-scores-section {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.kapoot-scores-section h3 {
    margin: 0 0 20px 0;
    color: #333;
}

.kapoot-teams-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}

.kapoot-team-card {
    background: rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 2px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}

.kapoot-team-card.submitted {
    border-color: #4CAF50;
    background: rgba(76, 175, 80, 0.1);
}

.kapoot-team-card.waiting {
    border-color: #FF9800;
    background: rgba(255, 152, 0, 0.1);
}

.kapoot-team-name {
    font-weight: bold;
    font-size: 16px;
    margin-bottom: 8px;
}

.kapoot-team-status {
    font-size: 14px;
    margin-bottom: 5px;
    opacity: 0.8;
}

.kapoot-team-score {
    font-size: 18px;
    font-weight: bold;
    color: #4CAF50;
}

/* Results Section */
.kapoot-results-section {
    text-align: center;
}

.kapoot-results-header h3 {
    margin: 0 0 30px 0;
    font-size: 28px;
    color: #4CAF50;
}

.kapoot-results-content {
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.kapoot-results-question {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
}

.kapoot-results-question h4 {
    margin: 0 0 10px 0;
    color: #333;
}

.kapoot-results-question p {
    margin: 0;
    font-size: 18px;
    line-height: 1.4;
}

.kapoot-results-teams h4 {
    margin: 0 0 15px 0;
    color: #333;
}

.kapoot-results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
}

.kapoot-result-team {
    background: rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.kapoot-result-name {
    font-weight: bold;
}

.kapoot-result-score {
    color: #4CAF50;
    font-weight: bold;
}

.kapoot-result-time {
    color: #666;
    font-size: 14px;
}

/* Final Results */
.kapoot-final-section {
    text-align: center;
}

.kapoot-final-header h2 {
    margin: 0 0 10px 0;
    font-size: 32px;
    color: #FFD700;
}

.kapoot-final-header p {
    margin: 0 0 30px 0;
    font-size: 18px;
    color: #555;
}

.kapoot-final-summary {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 30px;
}

.kapoot-final-rankings h3 {
    margin: 0 0 20px 0;
    color: #333;
}

.kapoot-final-grid {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.kapoot-final-team {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.kapoot-final-team.winner {
    background: linear-gradient(135deg, #FFD700, #FFA000);
    border-color: #FFD700;
    color: #000;
    font-weight: bold;
}

.kapoot-final-rank {
    font-size: 20px;
    font-weight: bold;
    min-width: 60px;
}

.kapoot-final-name {
    flex: 1;
    text-align: left;
    margin-left: 20px;
    font-size: 18px;
}

.kapoot-final-score {
    font-size: 20px;
    font-weight: bold;
    color: #4CAF50;
}

.kapoot-final-team.winner .kapoot-final-score {
    color: #000;
}

/* Modal Footer */
.kapoot-modal-footer {
    padding: 20px 30px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0 0 20px 20px;
}

.kapoot-modal-actions {
    display: flex;
    justify-content: center;
    gap: 15px;
}

.kapoot-btn {
    padding: 12px 24px;
    border: none;
    border-radius: 25px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    min-width: 120px;
}

.kapoot-btn.primary {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: #333;
}

.kapoot-btn.primary:hover {
    background: linear-gradient(135deg, #45a049, #3d8b40);
    transform: translateY(-2px);
}

.kapoot-btn.secondary {
    background: rgba(255, 255, 255, 0.1);
    color: #333;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

.kapoot-btn.secondary:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.5);
}

/* Responsive Design */
@media (max-width: 768px) {
    .kapoot-modal-content {
        width: 95%;
        margin: 20px;
    }
    
    .kapoot-modal-body {
        padding: 20px;
    }
    
    .kapoot-choices-container {
        grid-template-columns: 1fr;
    }
    
    .kapoot-quiz-header {
        flex-direction: column;
        gap: 20px;
        text-align: center;
    }
    
    .kapoot-timer-display {
        justify-content: center;
    }
    
    .kapoot-teams-grid {
        grid-template-columns: 1fr;
    }
    
    .kapoot-results-grid {
        grid-template-columns: 1fr;
    }
    
    .kapoot-modal-actions {
        flex-direction: column;
    }
    
    .kapoot-btn {
        width: 100%;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .kapoot-modal-content {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
}

`;class H{constructor(e,t,o,s,n=null){this.teams=e,this.webrtcChat=t,this.collaborativeWidget=o,this.shadowRoot=s,this.currentUserTeam=n,this.gameState="waiting",this.roundNumber=0,this.maxRounds=3,this.currentQuiz=null,this.currentTeam=null,this.currentQuizNumber=0,this.timer=new I(3e4),this.quiz=new D(t.quizRoomManager),this.teamScores={},this.initializeTeamScores(),this.setupTeamAnswerListener(),this.quizStructure={1:{timeLimit:30,showCountdown:!0,showProgressBar:!0,showExplanations:!1,teamBasedTimer:!1},2:{timeLimit:30,showCountdown:!1,showProgressBar:!1,showExplanations:!0,teamBasedTimer:!1},3:{timeLimit:30,showCountdown:!1,showProgressBar:!1,showExplanations:!0,teamBasedTimer:!1}},this.setupEventListeners(),this.injectStyles(),console.log("🎮 KapootGame initialized with teams:",e.map(i=>i.name))}initializeTeamScores(){this.teams.forEach(e=>{this.teamScores[e.name]={name:e.name,totalScore:0,roundScores:[],responseTime:null,lastAnswer:null}})}setupTeamAnswerListener(){document.addEventListener("kapootTeamAnswered",e=>{const{teamName:t}=e.detail;console.log("🎮 KapootGame received team answer event:",t),this.handleTeamAnswer(t)})}setupEventListeners(){document.addEventListener("kapootTeamResponse",e=>{this.handleTeamResponse(e.detail)}),document.addEventListener("kapootTimerExpiry",e=>{this.handleTimeUp()}),document.addEventListener("kapootExplanationReveal",e=>{this.handleExplanationsRevealed(e.detail)}),this.timer.setCallbacks({onTick:e=>{this.updateTimerDisplay(e)},onComplete:()=>{this.handleTimeUp()}})}injectStyles(){const e=this.shadowRoot.getElementById("kapoot-styles");e&&e.remove();const t=this.shadowRoot.ownerDocument.createElement("style");t.id="kapoot-styles",t.textContent=N,this.shadowRoot.appendChild(t)}setCollectedContent(e){this.collectedContent=e,console.log("🎮 Kapoot game received collected content:",{userCount:e?.userCount||0,totalCharacters:e?.totalCharacters||0,urls:e?.content?.map(t=>t.url)||[]})}async startGame(){console.log("🎮 Starting Kapoot game..."),this.gameState="active",this.roundNumber=0,this.broadcastGameState("game_started"),console.log("🎮 Starting first quiz question..."),await this.nextQuestion()}async nextQuestion(){if(this.gameState==="active"){if(this.currentQuizNumber++,this.currentQuizNumber>this.maxRounds){this.endGame();return}console.log(`🎮 Starting quiz ${this.currentQuizNumber}/${this.maxRounds}`);try{const e=this.quizStructure[this.currentQuizNumber];console.log("🎮 Quiz config for quiz",this.currentQuizNumber,":",e),console.log("🎮 Generating quiz from collected content..."),this.currentQuiz=await this.quiz.generateFromRoomContent(this.teams,this.collectedContent),console.log("🎮 Quiz generated successfully:",this.currentQuiz?.quizId),this.currentQuiz.quizNumber=this.currentQuizNumber,this.currentQuiz.totalQuizzes=this.maxRounds,this.currentQuizNumber===1&&(this.currentQuiz.specialFeatures={showCountdown:!0,showProgressBar:!0,showExplanations:!1,teamBasedTimer:!0}),this.resetTeamResponses();const t=this.quiz.createMCQuizMessage(this.currentQuiz,this.currentUserTeam);this.sendQuizMessage(t),this.broadcastGameState("quiz_started",this.currentQuiz),this.startQuizTimer(e)}catch(e){console.error("🎮 Failed to generate quiz:",e),this.handleQuizError()}}}startQuizTimer(e){this.quizStartTime=Date.now(),this.startUnifiedTimer(),console.log("🎮 Quiz timer started")}startUnifiedTimer(){this.teamCountdownInterval&&clearInterval(this.teamCountdownInterval),this.timer&&this.timer.stopTimer(),this.timer=new I(3e4),this.timer.callbacks.onTick=e=>{this.updateUnifiedTimer(e)},this.timer.callbacks.onComplete=()=>{this.handleUnifiedTimerExpiry()},this.timer.startTimer(),console.log("🎮 Unified timer started: 30 seconds for all teams")}updateUnifiedTimer(e){const t=Math.ceil(e/1e3);t!==this.lastLoggedSecond&&(console.log(`🎮 Timer update: ${t} seconds remaining`),this.lastLoggedSecond=t);const o=document.querySelector(".quiz-kapoot-timer");o&&(o.textContent=`${t}s ⏰`)}handleUnifiedTimerExpiry(){console.log("🎮 Unified timer expired"),this.showExplanations()}startTeamBasedTimer(){this.teamTimer=30,this.currentTeamIndex=0,this.startTeamTimer()}startTeamTimer(){const e=this.teams[this.currentTeamIndex];this.updateTeamCountdown(e.name,this.teamTimer),this.teamCountdownInterval=setInterval(()=>{this.teamTimer--,this.updateTeamCountdown(e.name,this.teamTimer),this.teamTimer<=0&&this.nextTeam()},1e3)}handleTeamAnswer(e){console.log("🎮 Team answered:",e),this.teamCountdownInterval&&(clearInterval(this.teamCountdownInterval),this.teamCountdownInterval=null),this.nextTeam()}nextTeam(){clearInterval(this.teamCountdownInterval),this.currentTeamIndex++,this.currentTeamIndex<this.teams.length?(this.teamTimer=30,this.startTeamTimer()):this.showExplanations()}startStandardTimer(){this.timer.startTimer()}updateTeamCountdown(e,t){console.log(`🎮 ${e}: ${t} seconds remaining`)}updateTimerDisplay(e){console.log("🎮 Timer update:",Math.ceil(e/1e3),"seconds remaining")}sendQuizMessage(e){this.collaborativeWidget&&(this.collaborativeWidget.addChatMessage("MC",e,!1),this.webrtcChat&&this.webrtcChat.broadcastMessage&&this.webrtcChat.broadcastMessage({type:"ai_response",payload:e,username:"MC"}))}showExplanations(){this.currentQuizNumber===1?this.displayExplanations():this.displayExplanations()}displayExplanations(){console.log("🎮 Showing explanations for quiz:",this.currentQuiz?.quizId),this.broadcastGameState("explanations_revealed",{quizId:this.currentQuiz?.quizId,explanations:this.currentQuiz?.explanations}),setTimeout(()=>{this.nextQuestion()},5e3)}handleExplanationsRevealed(e){console.log("🎮 Explanations revealed for quiz:",e.quizId)}handleTeamResponse(e){const{quizId:t,team:o,answer:s,responseTime:n,score:i}=e;if(!this.currentQuiz||this.currentQuiz.quizId!==t){console.warn("🎮 Received response for wrong quiz");return}const a=s===this.currentQuiz.correctAnswer;this.currentQuiz.teamResponses[o]={answered:!0,answer:s,responseTime:n,score:i,isCorrect:a},this.teamScores[o].totalScore+=i,this.teamScores[o].roundScores.push(i),this.teamScores[o].responseTime=n,this.teamScores[o].lastAnswer=s,console.log("🎮 Team response received:",{team:o,answer:s,responseTime:n,score:i,isCorrect:a}),this.broadcastGameState("team_response",{team:o,choice:s,responseTime:n,score:i,isCorrect:a}),this.checkAllTeamsResponded()}checkAllTeamsResponded(){this.teams.every(t=>this.currentQuiz.teamResponses[t.name]?.answered)&&(console.log("🎮 All teams have responded"),this.endQuizRound())}handleTimeUp(){console.log("🎮 Time up! Ending quiz round"),this.endQuizRound()}endQuizRound(){this.timer.stopTimer(),setTimeout(()=>{this.showRoundResults()},2e3)}showRoundResults(){const e={roundNumber:this.roundNumber,maxRounds:this.maxRounds,question:this.currentQuiz.question,teamScores:Object.values(this.teamScores).map(o=>({name:o.name,score:o.roundScores[o.roundScores.length-1]||0,responseTime:o.responseTime,totalScore:o.totalScore}))},t=this.createRoundResultsMessage(e);this.sendQuizMessage(t),this.broadcastGameState("round_results",e)}createRoundResultsMessage(e){const t=e.teamScores.map(o=>`${o.name}: ${o.score} pts (Total: ${o.totalScore} pts)`).join(`
`);return`🎮 **Round ${e.roundNumber} Results!**

**Question:** ${e.question}

**Team Scores:**
${t}

**Next Question:** ${e.roundNumber<e.maxRounds?"Starting soon...":"Game Complete!"}`}resetTeamResponses(){this.teams.forEach(e=>{this.currentQuiz.teamResponses[e.name]={answered:!1,answer:null,responseTime:null,score:0,isCorrect:!1}})}endGame(){console.log("🎮 Ending Kapoot game"),this.gameState="finished",this.timer.stopTimer(),this.showFinalResults(),this.broadcastGameState("game_ended",this.getFinalResults())}showFinalResults(){const e=this.getFinalResults(),t=this.createFinalResultsMessage(e);this.sendQuizMessage(t);const o=e.winner;this.sendSimpleMCMessage(`🏆 Kapoot game complete! Winner: ${o.name} with ${o.totalScore} points!`)}createFinalResultsMessage(e){return`🏆 **Kapoot Game Complete!**

**Final Rankings:**
${e.teamScores.map((o,s)=>`${s+1}${this.getRankSuffix(s+1)}. ${o.name}: ${o.totalScore} pts`).join(`
`)}

**Winner:** ${e.winner.name} with ${e.winner.totalScore} points!

**Game Stats:**
- Rounds Played: ${e.roundsPlayed}/${e.maxRounds}
- Total Teams: ${e.teamScores.length}

🎉 Congratulations to all teams!`}getFinalResults(){const e=Object.values(this.teamScores).sort((t,o)=>o.totalScore-t.totalScore);return{gameState:this.gameState,roundsPlayed:this.roundNumber,maxRounds:this.maxRounds,teamScores:e,winner:e[0]}}handleQuizError(){console.error("🎮 Quiz generation failed, ending game"),this.endGame()}broadcastGameState(e,t=null){if(!this.webrtcChat||!this.webrtcChat.broadcastMessage){console.warn("🎮 Cannot broadcast game state - WebRTC not available");return}const o={type:"kapoot_game_state",payload:{eventType:e,gameState:this.gameState,roundNumber:this.roundNumber,maxRounds:this.maxRounds,teamScores:this.teamScores,currentQuiz:this.currentQuiz,data:t,timestamp:Date.now()}};this.webrtcChat.broadcastMessage(o),console.log("🎮 Broadcasted game state:",e)}handleIncomingGameState(e){switch(console.log("🎮 Received game state:",e.eventType),e.eventType){case"quiz_started":this.displayQuizForPlayer(e.currentQuiz);break;case"team_response":this.updateTeamResponseDisplay(e.data);break;case"round_results":this.displayRoundResults(e.data);break;case"game_ended":this.displayFinalResults(e.data);break}}displayQuizForPlayer(e){const t=this.findCurrentUserTeam();t&&(this.currentQuiz=e,this.currentTeam=t.name,this.ui.createQuizDisplay(e,t.name),this.startQuizTimer())}findCurrentUserTeam(){const e=this.webrtcChat?.myUsername;return e?this.teams.find(t=>t.members.some(o=>o.username===e)):null}updateTeamResponseDisplay(e){this.currentQuiz&&(this.currentQuiz.teamResponses[e.team]={answer:e.choice,responseTime:e.responseTime,score:e.score,submitted:!0,isCorrect:e.isCorrect},this.ui.showTeamScores(this.currentQuiz.teamResponses))}displayRoundResults(e){e.teamScores.forEach(o=>{this.teamScores[o.name]&&(this.teamScores[o.name].totalScore=o.totalScore)});const t=this.ui.createGameResults(e);this.shadowRoot.appendChild(t)}displayFinalResults(e){this.gameState="finished",this.showFinalResults()}getRankSuffix(e){return e===1?"st":e===2?"nd":e===3?"rd":"th"}getGameState(){return{gameState:this.gameState,roundNumber:this.roundNumber,maxRounds:this.maxRounds,teamScores:this.teamScores,currentQuiz:this.currentQuiz}}sendSimpleMCMessage(e){this.collaborativeWidget&&(this.collaborativeWidget.addChatMessage("MC",e,!1),this.webrtcChat&&this.webrtcChat.broadcastMessage&&this.webrtcChat.broadcastMessage({type:"ai_response",payload:e,username:"MC"}))}cleanup(){this.timer.stopTimer(),this.gameState="finished",console.log("🎮 Kapoot game cleaned up")}}class E{constructor(e,t,o,s){this.shadowRoot=e,this.webrtcChat=t,this.gamificationButton=o,this.collaborativeWidget=s,this.teamManager=new P,this.kapootGame=null,this.panel=null,this.modal=null,this.isVisible=!1,this.isDragging=!1,this.offsetX=0,this.offsetY=0,this.currentUsersData=[],console.log("🎮 GamificationComponent created with webrtcChat:",this.webrtcChat),this.gameImages={"Kapoot!":"src/assets/kapoot.png","Capture the Flag!":"src/assets/flag.png",Remix:"src/assets/remix.png","Pac-Writer":"src/assets/pac-write.png"},this.gameDescriptions={"Kapoot!":"A fast-paced quiz game where players compete to answer questions correctly in the shortest amount of time. Points are awarded for speed and accuracy.","Capture the Flag!":"A team-based game where players try to find a hidden 'flag' (a specific piece of information or a location on the page) before the other teams.",Remix:"A creative game where players collaboratively rewrite or 'remix' a piece of text. The most creative and coherent remix wins.","Pac-Writer":"This is like pacman on a website, the pacman is assigned a keywords from the page and it aims to eat them... the goal is to eat as few non keywords (given) as possible while gaining points by eating the given keywords."},this.injectStyles(),this.createPanel()}updateWebRTCChat(e){console.log("🎮 Updating webrtcChat reference:",e),this.webrtcChat=e}injectStyles(){const e=this.shadowRoot.ownerDocument.createElement("style");e.textContent=U,this.shadowRoot.appendChild(e)}createPanel(){this.panel=this.shadowRoot.ownerDocument.createElement("div"),this.panel.id="gamification-panel",this.panel.className="gamification-panel";const e=Object.keys(this.gameDescriptions).map(o=>`
            <li>
                <img src="${this.gameImages[o]}" class="game-thumbnail">
                <div class="game-info">
                    <a href="#" class="game-title" data-game="${o}">${o}</a>
                    <p class="game-description">${this.gameDescriptions[o].substring(0,70)}...</p>
                </div>
            </li>
        `).join("");this.panel.innerHTML=`
            <div class="gamification-header">
                <span>Leaderboard</span>
                <button id="close-gamification-btn" title="Close">&times;</button>
            </div>
            <div class="gamification-tabs">
                <button class="tab-button active" data-tab="games">Games</button>
                <button class="tab-button" data-tab="players">Players</button>
                <button class="tab-button" data-tab="teams">Teams</button>
            </div>
            <div id="games-content" class="tab-content active">
                <ul class="games-list">${e}</ul>
            </div>
            <div id="players-content" class="tab-content">
                <table class="gamification-table">
                    <thead>
                        <tr>
                            <th>Participant</th>
                            <th>Location</th>
                            <th>Stars</th>
                            <th>Badges</th>
                            <th>Last Action</th>
                        </tr>
                    </thead>
                    <tbody id="gamification-players-body"></tbody>
                </table>
            </div>
            <div id="teams-content" class="tab-content">
                <table class="gamification-table">
                    <thead>
                        <tr>
                            <th>Team</th>
                            <th>Score</th>
                            <th>Members</th>
                            <th>Badges</th>
                        </tr>
                    </thead>
                    <tbody id="gamification-teams-body"></tbody>
                </table>
            </div>
        `,this.shadowRoot.appendChild(this.panel),this.modal=this.shadowRoot.ownerDocument.createElement("div"),this.modal.className="modal-overlay",this.modal.innerHTML=`
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="modal-title"></h3>
                    <button class="modal-close-btn">&times;</button>
                </div>
                <div class="modal-body">
                    <img id="modal-game-image" class="modal-game-image" src="">
                    <p id="modal-body-text"></p>
                </div>
                <div class="modal-footer">
                    <button id="configure-game-btn" class="configure-game-btn">Configure Game</button>
                </div>
            </div>
        `,this.shadowRoot.appendChild(this.modal),this.panel.querySelector(".gamification-header").addEventListener("mousedown",this.onMouseDown.bind(this)),this.shadowRoot.ownerDocument.addEventListener("mousemove",this.onMouseMove.bind(this)),this.shadowRoot.ownerDocument.addEventListener("mouseup",this.onMouseUp.bind(this)),this.panel.querySelector("#close-gamification-btn").addEventListener("click",()=>this.hide()),this.modal.querySelector(".modal-close-btn").addEventListener("click",()=>this.hideModal()),this.modal.querySelector("#configure-game-btn").addEventListener("click",()=>this.showTeamConfigurationModal()),this.modal.addEventListener("click",o=>{o.target===this.modal&&this.hideModal()}),this.panel.querySelectorAll(".tab-button").forEach(o=>{o.addEventListener("click",()=>{const s=o.dataset.tab;this.panel.querySelectorAll(".tab-button").forEach(n=>n.classList.remove("active")),o.classList.add("active"),this.panel.querySelectorAll(".tab-content").forEach(n=>n.classList.remove("active")),this.panel.querySelector(`#${s}-content`).classList.add("active")})}),this.panel.querySelectorAll(".games-list a").forEach(o=>{o.addEventListener("click",s=>{s.preventDefault(),this.showGameInfoModal(s.target.dataset.game)})})}showGameInfoModal(e){this.modal.querySelector("#modal-title").textContent=e,this.modal.querySelector("#modal-game-image").src=this.gameImages[e]||"",this.modal.querySelector("#modal-body-text").textContent=this.gameDescriptions[e]||"No description available.",this.modal.style.display="block"}hideModal(){this.modal.style.display="none"}showTeamConfigurationModal(){this.hideModal(),this.createTeamConfigurationModal()}getCurrentUsersData(){if(this.currentUsersData&&this.currentUsersData.length>0)return console.log("Using stored users data:",this.currentUsersData),this.currentUsersData;const e=[];return this.webrtcChat&&this.webrtcChat.myClientId&&this.webrtcChat.myUsername&&e.push({peerId:this.webrtcChat.myClientId,username:this.webrtcChat.myUsername,page:window.location.href,stars:0,totalScore:0,badges:[],lastAction:"Just joined"}),this.webrtcChat&&this.webrtcChat.peers&&Object.entries(this.webrtcChat.peers).forEach(([t,o])=>{o.username&&o.username!=="...joining..."&&e.push({peerId:t,username:o.username,page:window.location.href,stars:0,totalScore:0,badges:[],lastAction:"Connected"})}),e.length===0&&e.push({username:"Player 1",peerId:"test1",page:window.location.href,stars:0,totalScore:0,badges:[],lastAction:"Test"},{username:"Player 2",peerId:"test2",page:window.location.href,stars:0,totalScore:0,badges:[],lastAction:"Test"}),console.log("Fallback users data:",e),e}createTeamConfigurationModal(){const e=this.shadowRoot.getElementById("team-configuration-modal");e&&e.remove();const t=this.getCurrentUsersData(),o=this.modal.querySelector("#modal-title").textContent,s=this.modal.querySelector("#modal-body-text").textContent,n=this.shadowRoot.ownerDocument.createElement("div");n.id="team-configuration-modal",n.className="modal-overlay",n.style.display="block";const a=["Team Alpha","Team Better"].map((c,l)=>`
            <div class="team-section" data-team="${c}">
                <div class="team-header">
                    <input type="text" class="team-name-field" value="${c}" data-original="${c}">
                    <button class="remove-team-btn" data-team="${c}" style="display: none;">×</button>
                </div>
                <div class="team-players" data-team="${c}">
                    <!-- Players will be added here via drag and drop -->
                </div>
            </div>
        `).join(""),r=t.map(c=>`
            <div class="player-item" data-player="${c.username}" draggable="true">
                <span class="player-name">${c.username}</span>
            </div>
        `).join("");n.innerHTML=`
            <div class="modal-content team-configuration-modal">
                <div class="modal-header">
                    <h3>Configure ${o}</h3>
                    <button class="modal-close-btn">&times;</button>
                </div>
                <div class="modal-body">
                    <p class="game-description">${s}</p>
                    
                    <div class="drag-instructions">
                        <h4>Instructions</h4>
                        <div class="instruction-item">
                            <span class="instruction-icon">→</span>
                            <span>Drag players from "Unassigned Players" to any team</span>
                        </div>
                        <div class="instruction-item">
                            <span class="instruction-icon">↔</span>
                            <span>Drag players between teams to reassign them</span>
                        </div>
                        <div class="instruction-item">
                            <span class="instruction-icon">✏</span>
                            <span>Click team names to rename them</span>
                        </div>
                        <div class="instruction-item">
                            <span class="instruction-icon">+</span>
                            <span>Click "Add Team" to create more teams</span>
                        </div>
                    </div>
                    
                    <div class="teams-container">
                        <div class="teams-grid">
                            ${a}
                        </div>
                        <button id="add-team-btn" class="add-team-btn">+ Add Team</button>
                    </div>
                    
                    <div class="unassigned-players-section">
                        <h4>Unassigned Players</h4>
                        <div class="unassigned-players">
                            ${r}
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button id="start-game-btn" class="start-game-btn">Start Game</button>
                </div>
            </div>
        `,this.shadowRoot.appendChild(n),this.autoAssignPlayersToTeams(t,n),n.querySelector(".modal-close-btn").addEventListener("click",()=>{n.remove()}),n.addEventListener("click",c=>{c.target===n&&n.remove()}),n.querySelector("#start-game-btn").addEventListener("click",()=>{this.startGameWithCountdown(),n.remove()}),n.querySelector("#add-team-btn").addEventListener("click",()=>{this.addNewTeam(n)}),n.querySelectorAll(".team-name-field").forEach(c=>{c.addEventListener("change",()=>{this.updateTeamNames(n)})}),this.setupDragAndDrop(n)}autoAssignPlayersToTeams(e,t){const o=t.querySelectorAll(".team-section"),s=t.querySelector(".unassigned-players");s.innerHTML="",e.forEach((n,i)=>{const a=i%o.length,c=o[a].querySelector(".team-players"),l=this.shadowRoot.ownerDocument.createElement("div");l.className="player-item",l.draggable=!0,l.dataset.player=n.username,l.innerHTML=`<span class="player-name">${n.username}</span>`,l.addEventListener("dragstart",d=>{d.target.classList.add("dragging")}),l.addEventListener("dragend",d=>{d.target.classList.remove("dragging")}),c.appendChild(l)})}addNewTeam(e){const t=e.querySelector(".teams-grid"),o=t.children.length,s=`Team ${String.fromCharCode(67+o)}`,n=`
            <div class="team-section" data-team="${s}">
                <div class="team-header">
                    <input type="text" class="team-name-field" value="${s}" data-original="${s}">
                    <button class="remove-team-btn" data-team="${s}">×</button>
                </div>
                <div class="team-players" data-team="${s}">
                    <!-- Players will be added here via drag and drop -->
                </div>
            </div>
        `;t.insertAdjacentHTML("beforeend",n);const i=t.lastElementChild,a=i.querySelector(".team-name-field"),r=i.querySelector(".remove-team-btn");a.addEventListener("change",()=>{this.updateTeamNames(e)}),r.addEventListener("click",()=>{this.removeTeam(e,s)}),this.setupTeamDragAndDrop(i)}removeTeam(e,t){const o=e.querySelector(`[data-team="${t}"]`);if(o){const s=o.querySelectorAll(".player-item"),n=e.querySelector(".unassigned-players");s.forEach(i=>{n.appendChild(i)}),o.remove()}}updateTeamNames(e){e.querySelectorAll(".team-name-field").forEach(o=>{const s=o.value||o.dataset.original;o.dataset.original;const n=o.closest(".team-section");n&&(n.dataset.team=s,n.querySelector(".team-players").dataset.team=s)})}setupDragAndDrop(e){e.querySelectorAll(".team-section").forEach(o=>{this.setupTeamDragAndDrop(o)});const t=e.querySelector(".unassigned-players");this.setupUnassignedDragAndDrop(t)}setupTeamDragAndDrop(e){const t=e.querySelector(".team-players");t.addEventListener("dragover",o=>{o.preventDefault(),t.classList.add("drag-over")}),t.addEventListener("dragleave",o=>{t.contains(o.relatedTarget)||t.classList.remove("drag-over")}),t.addEventListener("drop",o=>{o.preventDefault(),t.classList.remove("drag-over");const s=this.shadowRoot.querySelector(".dragging");s&&(s.remove(),t.appendChild(s),s.classList.remove("dragging"),console.log(`Moved player to team: ${e.dataset.team}`))})}setupUnassignedDragAndDrop(e){e.addEventListener("dragover",t=>{t.preventDefault(),e.classList.add("drag-over")}),e.addEventListener("dragleave",t=>{e.contains(t.relatedTarget)||e.classList.remove("drag-over")}),e.addEventListener("drop",t=>{t.preventDefault(),e.classList.remove("drag-over");const o=this.shadowRoot.querySelector(".dragging");o&&(o.remove(),e.appendChild(o),o.classList.remove("dragging"),console.log("Moved player to unassigned"))}),e.addEventListener("dragstart",t=>{t.target.classList.contains("player-item")&&t.target.classList.add("dragging")}),e.addEventListener("dragend",t=>{t.target.classList.remove("dragging")})}startGameWithCountdown(){this.broadcastCountdownStart(),this.createCountdownModal()}broadcastCountdownStart(){let e="Player";if(this.currentUsersData&&this.currentUsersData.length>0){const o=this.webrtcChat?this.webrtcChat.myClientId:null;e=(this.currentUsersData.find(n=>n.peerId===o)||this.currentUsersData[0]).username}else this.webrtcChat&&this.webrtcChat.myUsername&&(e=this.webrtcChat.myUsername);const t=this.modal.querySelector("#modal-title").textContent;if(console.log("🎮 Checking webrtcChat availability:",{webrtcChat:!!this.webrtcChat,broadcastMessage:!!(this.webrtcChat&&this.webrtcChat.broadcastMessage),webrtcChatType:typeof this.webrtcChat}),this.webrtcChat&&this.webrtcChat.broadcastMessage){const o={type:"countdown_start",payload:{initiator:e,gameTitle:t,timestamp:Date.now()}};console.log("🎮 Broadcasting countdown message:",o),this.webrtcChat.broadcastMessage(o)}else console.log("❌ Cannot broadcast countdown - webrtcChat not available"),console.log("❌ webrtcChat:",this.webrtcChat),console.log("❌ broadcastMessage method:",this.webrtcChat?.broadcastMessage);console.log(`Broadcasting countdown start: ${e} initiated ${t}`)}handleCountdownStart(e){if(console.log("🎮 GamificationComponent received countdown start:",e),console.log("🎮 Current user:",this.webrtcChat?.myUsername),console.log("🎮 Initiator:",e.initiator),e.initiator===this.webrtcChat?.myUsername||e.initiator===this.webrtcChat?.myClientId){console.log("🎮 Skipping countdown - we are the initiator");return}console.log(`🎮 Creating countdown modal for ${e.initiator} starting ${e.gameTitle}`),this.createCountdownModal(e.initiator,e.gameTitle)}sendGameRulesMessage(){const e=this.modal.querySelector("#modal-title").textContent,t=this.modal.querySelector("#modal-body-text").textContent,o=this.modal.querySelector("#modal-game-image").src;if(this.collaborativeWidget){const s=this.getGameRules(e),n=`🎮 **${e}** – ${t}

<img src="${o}" alt="${e}" style="max-width: 200px; height: auto; border-radius: 8px; margin: 10px 0;">

${s}`;this.collaborativeWidget.addChatMessage("MC",n,!1),this.webrtcChat&&this.webrtcChat.broadcastMessage&&this.webrtcChat.broadcastMessage({type:"ai_response",payload:n,username:"MC"})}console.log(`Game started: ${e} - MC message sent directly`)}getGameRules(e){return{"Kapoot!":`## Element  How It Works
**Objective**  Be the first to answer questions correctly and as fast as possible.
**Setup**  1‑2+ players (or a team) join a Kapoot! room. A host (you or a designated player) loads a question set.
**Gameplay**  • Each question appears on a shared screen.
• Players tap their answer button as soon as they're sure.
• The time taken to press the button matters! Faster answers get more points.
• Correct answers earn points based on speed.
• Wrong answers deduct points.
• The player with the most points wins!`,"Capture the Flag!":`## Element  How It Works
**Objective**  Find the hidden 'flag' (specific information) before other teams.
**Setup**  Teams are assigned and given clues about the flag location.
**Gameplay**  • Teams search for the hidden flag using provided clues.
• First team to find and capture the flag wins.
• Teams can collaborate and strategize together.
• Use teamwork to solve puzzles and find the flag!`,Remix:`## Element  How It Works
**Objective**  Create the most creative and coherent 'remix' of given text.
**Setup**  Players are given original text to remix creatively.
**Gameplay**  • Players collaboratively rewrite or 'remix' the text.
• Focus on creativity, humor, and coherence.
• Teams vote on the best remix.
• Most creative and well-written remix wins!`,"Pac-Writer":`## Element  How It Works
**Objective**  Navigate like Pac-Man to collect keywords while avoiding non-keywords.
**Setup**  Players are assigned keywords to collect from the page.
**Gameplay**  • Move around the page like Pac-Man.
• Collect assigned keywords for points.
• Avoid non-keywords (they reduce points).
• Strategy: plan your path to maximize keyword collection.
• Player with highest score wins!`}[e]||`## Game Rules
**Objective**  Follow the game instructions and compete with other players.
**Setup**  Join the game room and wait for instructions.
**Gameplay**  • Follow the game-specific rules.
• Work with your team if applicable.
• Have fun and compete fairly!`}createCountdownModal(e=null,t=null){const o=this.shadowRoot.getElementById("countdown-modal");o&&o.remove();const s=this.shadowRoot.ownerDocument.createElement("div");s.id="countdown-modal",s.className="modal-overlay countdown-overlay",s.style.display="block";let n=e||"Player";if(!e)if(this.currentUsersData&&this.currentUsersData.length>0){const d=this.webrtcChat?this.webrtcChat.myClientId:null;n=(this.currentUsersData.find(m=>m.peerId===d)||this.currentUsersData[0]).username,console.log("Found current user from stored data:",n,"myPeerId:",d)}else this.webrtcChat&&this.webrtcChat.myUsername?(n=this.webrtcChat.myUsername,console.log("Found current user from webrtcChat:",n)):console.log("Using fallback username: Player");const i=t||this.modal.querySelector("#modal-title").textContent;s.innerHTML=`
            <div class="countdown-content">
                <div class="countdown-header">
                    <h2 class="countdown-title">${n} has initiated ${i}!</h2>
                    <p class="countdown-subtitle">Start-time in:</p>
                </div>
                <pre id="ascii-display" class="ascii-display"></pre>
            </div>
        `,this.shadowRoot.appendChild(s);const a=[`
███████╗
██╔════╝
███████╗
╚════██║
███████║
╚══════╝
            `,`
██   ██╗
██   ██║
███████║
     ██║
     ██║
     ╚═╝
            `,`
██████╗ 
╚═══██║
██████║
╚═══██║
██████║
╚═════╝ 
            `,`
██████╗ 
██ ╔██║
   ██╔╝ 
  ██╔╝  
███████╗
╚══════╝
            `,`
  ██╗
 ███║
 ╚██║
  ██║
  ██║
  ╚═╝
            `,`
 ██████╗   ██████╗  ██╗
██╔════╝  ██╔═══██╗ ██║
██║  ███╗ ██║   ██║ ██║
██║   ██║ ██║   ██║ ╚═╝
╚██████╔╝  ██████╗  ██╗
 ╚═════╝   ╚═════╝  ╚═╝
            `];let r=0;const c=s.querySelector("#ascii-display");c.textContent=a[r];const l=setInterval(()=>{r++,r<a.length?c.textContent=a[r]:(clearInterval(l),setTimeout(()=>{s.remove(),this.triggerImmediateContentCollection()},1e3))},1e3)}async triggerImmediateContentCollection(){console.log("🎮 Countdown completed - triggering immediate content collection");try{if(this.webrtcChat&&this.webrtcChat.quizRoomManager){const e=await this.webrtcChat.quizRoomManager.startImmediateContentCollection();this.collectedContent=e,console.log("🎮 Content collection completed:",e),this.showGameInstructions(),setTimeout(()=>{this.startKapootGame()},2e3)}else console.warn("🎮 Quiz room manager not available for content collection"),this.showGameInstructions(),setTimeout(()=>{this.startKapootGame()},2e3)}catch(e){console.error("🎮 Content collection failed:",e),this.showGameInstructions(),setTimeout(()=>{this.startKapootGame()},2e3)}}showGameInstructions(){const e=`🎮 **Welcome to Kapoot!**

![Kapoot Game](src/assets/kapoot.png)

**Game Rules:**
- 3 quiz questions total
- Each team gets 30 seconds to answer
- Fast answers get more points
- Work together to win!

**Content Collected:** ${this.collectedContent?`${this.collectedContent.userCount} users' page content`:"No content available"}

**Ready to start?** The first quiz will begin shortly...`;this.collaborativeWidget?(console.log("🎮 Adding MC message to chat:",e),this.collaborativeWidget.addChatMessage("MC",e,!1),this.webrtcChat&&this.webrtcChat.broadcastMessage&&this.webrtcChat.broadcastMessage({type:"ai_response",payload:e,username:"MC"})):console.error("🎮 Collaborative widget not available for MC message")}startKapootGame(){if(console.log("🎮 Starting Kapoot game with collected content:",this.collectedContent),!this.kapootGame){console.log("🎮 Initializing KapootGame...");const e=this.teamManager.getTeams(this.currentUsersData),t=this.findCurrentUserTeam(e);this.kapootGame=new H(e,this.webrtcChat,this.collaborativeWidget,this.shadowRoot,t),console.log("🎮 KapootGame initialized with teams:",e,"current user team:",t)}this.kapootGame?(this.kapootGame.setCollectedContent(this.collectedContent),this.kapootGame.startGame(),console.log("🎮 Kapoot game started successfully")):console.error("🎮 Failed to initialize Kapoot game")}findCurrentUserTeam(e){const t=this.webrtcChat?.myClientId;if(!t)return null;for(const o of e)if(o.members&&o.members.some(s=>s.peerId===t))return o.name;return e.length>0?e[0].name:null}onMouseDown(e){e.target.closest("button")||(this.isDragging=!0,this.offsetX=e.clientX-this.panel.offsetLeft,this.offsetY=e.clientY-this.panel.offsetTop,e.preventDefault())}onMouseMove(e){this.isDragging&&(this.panel.style.left=`${e.clientX-this.offsetX}px`,this.panel.style.top=`${e.clientY-this.offsetY}px`)}onMouseUp(){this.isDragging=!1}toggle(){this.isVisible?this.hide():this.show()}show(){this.panel.style.display="block",this.isVisible=!0,this.update([])}hide(){this.panel.style.display="none",this.isVisible=!1}update(e=[]){if(this.currentUsersData=e,!this.isVisible)return;this.updatePlayersView(e);const t=this.teamManager.getTeams(e);this.updateTeamsView(t)}updatePlayersView(e){const t=this.panel.querySelector("#gamification-players-body");t.innerHTML="",e.forEach(o=>{const s=this.shadowRoot.ownerDocument.createElement("tr"),n=o.page?`...${o.page.slice(-20)}`:"Unknown",i=o.page?`<a href="${o.page}" target="_blank">${n}</a>`:"Unknown",a=(o.badges||[]).map(c=>`<span class="badge">${c.emoji||c.icon}<span class="tooltip">${c.name||c.text}</span></span>`).join(" ");o.totalScore||o.stars;const r=o.stars||o.totalScore||0;o.stars!==o.totalScore&&console.warn(`Stars/Score mismatch for ${o.username}: stars=${o.stars}, totalScore=${o.totalScore}`),s.innerHTML=`
                <td>${o.username}</td>
                <td>${i}</td>
                <td>
                    <div class="star-display">
                        <span class="star-icon">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                        </span>
                        <span>${r}</span>
                    </div>
                </td>
                <td>${a}</td>
                <td>${o.lastAction||""}</td>
            `,t.appendChild(s)})}updateTeamsView(e){const t=this.panel.querySelector("#gamification-teams-body");t.innerHTML="",e.forEach(o=>{const s=this.shadowRoot.ownerDocument.createElement("tr"),n=o.members.map(a=>`<li>${a.username}</li>`).join(""),i=o.badges.map(a=>`<span class="badge">${a.icon}<span class="tooltip">${a.text}</span></span>`).join(" ");s.innerHTML=`
                <td>${o.name}</td>
                <td>${o.score}</td>
                <td><ul class="team-members-list">${n}</ul></td>
                <td>${i}</td>
            `,t.appendChild(s)})}}class F{constructor(e){this.shadowRoot=e,this.toastContainer=null,this.createToastContainer()}createToastContainer(){this.toastContainer=document.createElement("div"),this.toastContainer.id="toast-container",this.toastContainer.className="toast-container",this.shadowRoot.appendChild(this.toastContainer)}showPointAward(e,t,o){const s=document.createElement("div");s.className="toast toast-point-award",s.innerHTML=`
            <div class="toast-content">
                <div class="toast-icon">⭐</div>
                <div class="toast-text">
                    <div class="toast-title">${e} earned ${t} star${t>1?"s":""}!</div>
                    <div class="toast-subtitle">${o}</div>
                </div>
                <div class="toast-sparkles">
                    <span class="sparkle">✨</span>
                    <span class="sparkle">✨</span>
                    <span class="sparkle">✨</span>
                </div>
            </div>
        `,this.showToast(s,4e3)}showQuizSubmission(e,t){const o=document.createElement("div");o.className=`toast toast-quiz-submission ${t?"correct":"incorrect"}`;const s=t?"🎉":"😔",n=t?"Correct Answer!":"Incorrect Answer";o.innerHTML=`
            <div class="toast-content">
                <div class="toast-icon">${s}</div>
                <div class="toast-text">
                    <div class="toast-title">${e}</div>
                    <div class="toast-subtitle">${n}</div>
                </div>
            </div>
        `,this.showToast(o,3e3)}showNotification(e,t="info"){const o=document.createElement("div");o.className=`toast toast-notification ${t}`;const s={success:"✅",error:"❌",info:"ℹ️",warning:"⚠️"};o.innerHTML=`
            <div class="toast-content">
                <div class="toast-icon">${s[t]||s.info}</div>
                <div class="toast-text">
                    <div class="toast-title">${e}</div>
                </div>
            </div>
        `,this.showToast(o,3e3)}showToast(e,t=3e3){this.toastContainer.appendChild(e),setTimeout(()=>{e.classList.add("show")},10),setTimeout(()=>{e.classList.add("hide"),setTimeout(()=>{e.parentNode&&e.parentNode.removeChild(e)},300)},t)}clearAll(){this.toastContainer&&(this.toastContainer.innerHTML="")}}class C{constructor(e){this.widget=e,this.activeQuizzes=new Map,this.quizSubmissions=new Map,this.toastNotifications=new F(e.shadowRoot)}setupQuizInteractions(e){e.querySelectorAll(".message-quiz-component, .quiz-kapoot-active").forEach(o=>{const s=o.dataset.quizId;if(s){if(o.dataset.quizSetup==="true"){console.log(`Quiz ${s} already set up, skipping...`);return}if(o.classList.contains("quiz-kapoot-active")){this.setupKapootQuizInteractions(o),o.dataset.quizSetup="true";return}if(this.activeQuizzes.has(s)){console.log(`Quiz ${s} state already exists, skipping setup...`);return}o.dataset.quizSetup="true",this.activeQuizzes.set(s,{isSubmitted:!1,selectedChoice:null,correctAnswer:o.dataset.correctAnswer}),this.quizSubmissions.has(s)||this.quizSubmissions.set(s,[]),this.setupQuizComponent(o),this.updateQuizDisplayWithSubmissions(s,o)}})}setupKapootQuizInteractions(e){const t=e.dataset.quizId;console.log("🎮 Setting up Kapoot quiz interactions for:",t),console.log("🎮 Quiz component:",e),console.log("🎮 Quiz data attribute:",e.dataset.quizData);const o=e.querySelectorAll(".quiz-kapoot-choice");console.log("🎮 Found choice buttons:",o.length),o.forEach((s,n)=>{console.log(`🎮 Setting up button ${n}:`,s),s.addEventListener("click",i=>{i.preventDefault(),i.stopPropagation();const a=i.currentTarget.dataset.choice;console.log("🎮 Kapoot choice clicked:",a),console.log("🎮 Event target:",i.currentTarget),this.handleKapootTeamAnswer(e,a)})}),console.log("🎮 Kapoot quiz interactions set up for:",t)}handleKapootTeamAnswer(e,t){const o=e.dataset.quizId;console.log("🎮 Handling Kapoot team answer:",{quizId:o,choice:t}),console.log("🎮 Quiz component in handler:",e);const s=this.extractKapootQuizData(e);if(console.log("🎮 Extracted quiz data:",s),!s){console.error("🎮 Could not extract quiz data from component");return}const n=s.currentUserTeam;if(!n){console.warn("🎮 No team found for current user");return}if(s.teamResponses[n].answered){console.log("🎮 Team already answered:",n),this.toastNotifications.showNotification("Your team has already answered!","info");return}const i=Date.now()-(s.startTime||Date.now()),a=this.calculateKapootScore(i,s.basePoints,t===s.correctAnswer);s.teamResponses[n]={answered:!0,answer:t,responseTime:i,score:a,isCorrect:t===s.correctAnswer},console.log("🎮 Team answer processed:",{team:n,choice:t,score:a,isCorrect:t===s.correctAnswer}),this.updateQuizDataInComponent(e,s),this.updateKapootQuizDisplay(e,s);const r=t===s.correctAnswer?`✅ Correct! Your team scored ${a} points!`:`❌ Incorrect. The correct answer is ${s.correctAnswer}.`;this.toastNotifications.showNotification(r,t===s.correctAnswer?"success":"error"),Object.values(s.teamResponses).every(l=>l.answered)&&(console.log("🎮 All teams answered, revealing explanations early"),this.revealKapootExplanations(e,s)),this.broadcastKapootTeamResponse(s,n,t,i,a),this.notifyKapootGameTeamAnswer(n)}extractKapootQuizData(e){try{const t=e.querySelector("script.quiz-data");if(t){const s=JSON.parse(t.textContent);return console.log("🎮 Extracted quiz data from script tag:",s),s}let o=e.dataset.quizData;if(console.log("🎮 Quiz data string from data attribute:",o),o){try{o=decodeURIComponent(o),console.log("🎮 URI decoded quiz data string:",o)}catch{console.log("🎮 Not URI encoded, trying HTML entity decoding"),o=o.replace(/&quot;/g,'"').replace(/&amp;/g,"&").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&#39;/g,"'").replace(/&apos;/g,"'")}console.log("🎮 Final decoded quiz data string:",o);const s=JSON.parse(o);return console.log("🎮 Extracted quiz data from data attribute:",s),s}return console.error("🎮 No quiz data found in component"),null}catch(t){return console.error("🎮 Error extracting quiz data:",t),console.error("🎮 Quiz component:",e),null}}calculateKapootScore(e,t,o){if(!o)return 0;const s=3e4-e,n=Math.max(0,s/3e4);return Math.floor(t*n)}updateQuizDataInComponent(e,t){const o=e.querySelector("script.quiz-data");o&&(o.textContent=JSON.stringify(t),console.log("🎮 Updated quiz data in script tag"))}updateKapootQuizDisplay(e,t){const o=e.querySelector(".quiz-kapoot-team-status");if(o){const n=Object.entries(t.teamResponses).map(([i,a])=>`${i}: ${a.answered?"✓ Answered":"⏳ Thinking"}`).join(" | ");o.innerHTML=`<strong>Teams:</strong> ${Object.keys(t.teamResponses).join(", ")}<br><strong>Status:</strong> ${n}`}e.querySelectorAll(".quiz-kapoot-choice").forEach(n=>{const i=n.dataset.choice,a=t.currentUserTeam;if(t.teamResponses[a].answered){const r=t.teamResponses[a].answer;n.disabled=!0,n.style.opacity="0.6",n.style.cursor="not-allowed",i===r&&(n.style.background="rgba(34, 197, 94, 0.3)",n.style.borderColor="rgba(34, 197, 94, 0.8)",n.style.color="#166534")}})}revealKapootExplanations(e,t){this.toastNotifications.showNotification("All teams answered! Explanations will be revealed.","info")}notifyKapootGameTeamAnswer(e){const t=new CustomEvent("kapootTeamAnswered",{detail:{teamName:e}});document.dispatchEvent(t),console.log("🎮 Dispatched team answer event for:",e)}broadcastKapootTeamResponse(e,t,o,s,n){this.widget.webrtcChat&&this.widget.webrtcChat.broadcastMessage&&this.widget.webrtcChat.broadcastMessage({type:"kapoot_team_response",payload:{quizId:e.quizId,team:t,answer:o,responseTime:s,score:n}})}setupQuizComponent(e){const t=e.dataset.quizId,o=e.querySelectorAll(".quiz-choice"),s=e.querySelector(".quiz-submit-btn");e.querySelector(".quiz-explanation"),e.querySelector(".quiz-result"),o.forEach(n=>{const i=n.querySelector(".quiz-radio");i.addEventListener("change",()=>{this.handleChoiceSelection(t,n,s)}),n.addEventListener("click",a=>{a.target!==i&&(i.checked=!0,i.dispatchEvent(new Event("change")))})}),s&&s.addEventListener("click",()=>{this.handleQuizSubmission(t,e)})}handleChoiceSelection(e,t,o){const s=this.activeQuizzes.get(e);!s||s.isSubmitted||(s.selectedChoice=t.dataset.choice,o&&(o.disabled=!1))}handleQuizSubmission(e,t){const o=this.activeQuizzes.get(e);if(!o||!o.selectedChoice||o.isSubmitted)return;o.isSubmitted=!0;const s=t.querySelectorAll(".quiz-choice"),n=t.querySelector(".quiz-submit-btn"),i=t.querySelector(".quiz-explanation"),a=t.querySelector(".quiz-result"),r=o.correctAnswer,c=o.selectedChoice;s.forEach(l=>l.style.pointerEvents="none"),n.disabled=!0,n.innerHTML=`
            <svg class="submit-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 12l2 2 4-4"/>
                <circle cx="12" cy="12" r="10"/>
            </svg>
            <span class="submit-text">Submitted</span>
        `,a&&this.showQuizResult(a,c,r),s.forEach(l=>{const d=l.dataset.choice,h=l.querySelector(".choice-text");if(d===r){const m=this.getExplanationForChoice(t,d);if(m){const u=document.createElement("div");u.className="choice-explanation correct",u.innerHTML=`
                        <svg class="explanation-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M9 12l2 2 4-4"/>
                            <circle cx="12" cy="12" r="10"/>
                        </svg>
                        <span class="explanation-text">${m}</span>
                    `,h.appendChild(u)}}else if(d===c){const m=this.getExplanationForChoice(t,d);if(m){const u=document.createElement("div");u.className="choice-explanation incorrect",u.innerHTML=`
                        <svg class="explanation-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="15" y1="9" x2="9" y2="15"/>
                            <line x1="9" y1="9" x2="15" y2="15"/>
                        </svg>
                        <span class="explanation-text">${m}</span>
                    `,h.appendChild(u)}}}),i&&(i.style.display="none"),this.trackQuizSubmission(e,c,r,!0),c===r&&(this.awardQuizPoints(),this.toastNotifications.showPointAward(this.widget.webrtcChat?.myUsername||"You",1,"Correct quiz answer!")),this.broadcastQuizSubmission(e,c,r),this.broadcastQuizLock(e,c,r)}showQuizResult(e,t,o){const s=t===o,n=e.querySelector(".result-text"),i=e.querySelector(".result-icon");s?(n.textContent="Correct!",i.innerHTML=`
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 12l2 2 4-4"/>
                    <circle cx="12" cy="12" r="10"/>
                </svg>
            `,e.style.borderColor="rgba(34, 197, 94, 0.6)",e.style.background="rgba(34, 197, 94, 0.1)"):(n.textContent="Incorrect. The correct answer was "+o,i.innerHTML=`
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="15" y1="9" x2="9" y2="15"/>
                    <line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
            `,e.style.borderColor="rgba(239, 68, 68, 0.6)",e.style.background="rgba(239, 68, 68, 0.1)"),e.style.display="flex"}getExplanationForChoice(e,t){const o=e.querySelector(".quiz-explanation");if(!o)return null;const s=o.querySelector(`[data-choice="${t}"]`);if(!s)return null;const n=s.querySelector(".explanation-text");return n?n.textContent:null}broadcastQuizSubmission(e,t,o){if(this.widget.webrtcChat){const s={type:"quiz_submission",quizId:e,selectedAnswer:t,correctAnswer:o,isCorrect:t===o,username:this.widget.webrtcChat.myUsername,timestamp:Date.now(),userId:this.widget.webrtcChat.myClientId};this.widget.webrtcChat.broadcastMessage(s)}}handleQuizSubmissionFromPeer(e){this.trackQuizSubmission(e.quizId,e.selectedAnswer,e.correctAnswer,!1,e.username);const t=this.widget.shadowRoot.querySelector(`[data-quiz-id="${e.quizId}"]`);if(!t)return;this.lockQuizAndShowResults(e.quizId,t,e);const o=e.isCorrect?"✓ Correct!":"✗ Incorrect",s=e.username||"Unknown User";this.widget.addSystemMessage(`${s} submitted: ${e.selectedAnswer} ${o}`),this.toastNotifications.showQuizSubmission(s,e.isCorrect)}trackQuizSubmission(e,t,o,s=!1,n=null){const i=this.quizSubmissions.get(e)||[],a=n||this.widget.webrtcChat?.myUsername||"Unknown",r=i.find(c=>c.username===a);r?(r.selectedAnswer=t,r.isCorrect=t===o,r.timestamp=Date.now()):i.push({username:a,selectedAnswer:t,correctAnswer:o,isCorrect:t===o,timestamp:Date.now(),isOwn:s}),this.quizSubmissions.set(e,i)}updateQuizDisplayWithSubmissions(e,t){const o=this.quizSubmissions.get(e)||[],s=t.querySelectorAll(".quiz-choice");s.forEach(n=>{const i=n.querySelector(".choice-text");i&&i.querySelectorAll(".submission-indicator").forEach(r=>r.remove())}),s.forEach(n=>{const i=n.dataset.choice,a=n.querySelector(".choice-text"),r=o.filter(c=>c.selectedAnswer===i);if(r.length>0&&a){const c=document.createElement("div");c.className="submission-indicator",c.innerHTML=r.map(l=>`<span class="submission-badge ${l.isCorrect?"correct":"incorrect"}" title="${l.username}">${l.username.charAt(0).toUpperCase()}</span>`).join(" "),a.appendChild(c)}})}awardQuizPoints(){this.widget&&this.widget.awardQuizPoint&&(this.widget.awardQuizPoint(this.widget.currentRoom),this.widget.gamificationComponent&&this.widget.updatePresenceList(),this.widget.addSystemMessage("🎉 Correct answer! You earned 1 point! ⭐"))}handleQuizAnswer(e){console.log("Quiz answer selected:",e),this.widget.webrtcChat&&this.widget.webrtcChat.broadcastMessage({type:"quiz_answer",quizId:e.quizId,choice:e.choice,userId:this.widget.webrtcChat.myClientId,username:this.widget.webrtcChat.myUsername,isCorrect:e.isCorrect,timestamp:e.timestamp})}cleanupQuiz(e){this.activeQuizzes.delete(e)}cleanupAllQuizzes(){this.activeQuizzes.clear()}broadcastQuizLock(e,t,o){if(this.widget.webrtcChat){const s={type:"quiz_lock",quizId:e,selectedAnswer:t,correctAnswer:o,username:this.widget.webrtcChat.myUsername,timestamp:Date.now(),userId:this.widget.webrtcChat.myClientId};this.widget.webrtcChat.broadcastMessage(s)}}lockQuizAndShowResults(e,t,o){const s=t.querySelectorAll(".quiz-choice"),n=t.querySelector(".quiz-submit-btn"),i=t.querySelector(".quiz-explanation"),a=t.querySelector(".quiz-result"),r=o.correctAnswer,c=o.selectedAnswer;if(s.forEach(l=>{l.style.pointerEvents="none",l.style.opacity="0.7"}),n&&(n.disabled=!0,n.innerHTML=`
                <svg class="submit-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 12l2 2 4-4"/>
                    <circle cx="12" cy="12" r="10"/>
                </svg>
                <span class="submit-text">Quiz Completed</span>
            `),a){const l=c===r,d=a.querySelector(".result-text"),h=a.querySelector(".result-icon");l?(d.textContent="Correct!",h.innerHTML=`
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 12l2 2 4-4"/>
                        <circle cx="12" cy="12" r="10"/>
                    </svg>
                `,a.style.borderColor="rgba(34, 197, 94, 0.6)",a.style.background="rgba(34, 197, 94, 0.1)"):(d.textContent="Incorrect. The correct answer was "+r,h.innerHTML=`
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="15" y1="9" x2="9" y2="15"/>
                        <line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                `,a.style.borderColor="rgba(239, 68, 68, 0.6)",a.style.background="rgba(239, 68, 68, 0.1)"),a.style.display="flex"}s.forEach(l=>{const d=l.dataset.choice,h=d===r,m=d===c;h?l.classList.add("correct"):l.classList.add("incorrect");const u=document.createElement("div");u.className=`choice-explanation ${h?"correct":"incorrect"}`;const b=this.getExplanationForChoice(t,d);h?u.innerHTML=`
                    <svg class="explanation-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 12l2 2 4-4"/>
                        <circle cx="12" cy="12" r="10"/>
                    </svg>
                    <span class="explanation-text">${b||"Correct answer"}</span>
                `:m?u.innerHTML=`
                    <svg class="explanation-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="15" y1="9" x2="9" y2="15"/>
                        <line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                    <span class="explanation-text">${b||"Incorrect answer"}</span>
                `:u.innerHTML=`
                    <svg class="explanation-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="15" y1="9" x2="9" y2="15"/>
                        <line x1="9" y1="9" x2="15" y2="15"/>
                    </svg>
                    <span class="explanation-text">${b||"Incorrect option"}</span>
                `;const f=l.parentNode,v=l.nextSibling;v?f.insertBefore(u,v):f.appendChild(u)}),i&&(i.style.display="none")}handleQuizLockFromPeer(e){const t=this.widget.shadowRoot.querySelector(`[data-quiz-id="${e.quizId}"]`);if(!t)return;this.lockQuizAndShowResults(e.quizId,t,e);const o=e.username||"Unknown User";this.widget.addSystemMessage(`Quiz completed by ${o}! Results are now visible.`)}}class ${constructor(){this.symbols=["@","#","$","%","&","*","+","=","?","!","~","^","|","\\","/","<",">","[","]","{","}","(",")",":",";",'"',"'","`"],this.animationDuration=800,this.roomAnimationDuration=1500,this.roomSymbolInterval=40,this.isAnimating=!1,this.lastAnimationTime=0,this.animationCooldown=2e3,this.lastStats=null,this.sessionGreeting=null,this.greetingMessages=["Welcome, {name}!","Hello, {name}!","What's up, {name}?","Hey {name}!","Howdy {name}!","You rock, {name}!","You're shining, {name}!","Awesome {name}!","Looking good, {name}!","Great to see you, {name}!","You're amazing, {name}!","Keep it up, {name}!","You're on fire, {name}!","Fantastic {name}!","You're the best, {name}!","Rock on, {name}!","You're incredible, {name}!","Way to go, {name}!","You're unstoppable, {name}!","Brilliant {name}!","You're killing it, {name}!","Outstanding {name}!","You're phenomenal, {name}!","Magnificent {name}!","You're spectacular, {name}!","Superb {name}!","You're extraordinary, {name}!","Exceptional {name}!","You're remarkable, {name}!","Incredible {name}!","You're fantastic, {name}!","Wonderful {name}!","You're brilliant, {name}!","Marvelous {name}!","You're outstanding, {name}!","Terrific {name}!","You're magnificent, {name}!","Splendid {name}!","You're superb, {name}!","Excellent {name}!","You're spectacular, {name}!","Perfect {name}!","You're extraordinary, {name}!","Amazing {name}!","You're exceptional, {name}!","Incredible {name}!","You're remarkable, {name}!","Fantastic {name}!","You're incredible, {name}!"]}getRandomGreeting(e){if(this.sessionGreeting)return this.sessionGreeting;const t=this.greetingMessages[Math.floor(Math.random()*this.greetingMessages.length)];return this.sessionGreeting=t.replace("{name}",e||"there"),console.log(`New session greeting selected: "${this.sessionGreeting}"`),this.sessionGreeting}updateStatsDisplay(e,t,o,s,n=null){if(!e){console.warn("Stats content element not found");return}const i=o&&t.roomScores[o]||0,a=s?"connected":"disconnected",r=o||"No Room";e.innerHTML=`
            <div class="stat-item room-status ${a}">
                <span class="stat-label">Room:</span>
                <span class="stat-value" id="room-name-display">${r}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Score:</span>
                <span class="stat-value" id="total-score-display">${t.totalScore}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Room Score:</span>
                <span class="stat-value" id="room-score-display">${i}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Messages:</span>
                <span class="stat-value" id="messages-display">${t.totalMessages}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Comments:</span>
                <span class="stat-value" id="comments-display">${t.totalComments}</span>
            </div>
        `}animateStatsDisplay(e,t,o,s,n=null){if(!e){console.warn("Stats content element not found");return}const i=Date.now(),a=JSON.stringify({userStats:t,currentRoom:o,isConnected:s});if(this.lastStats===a){console.log("Animation skipped - stats unchanged");return}if(this.isAnimating||i-this.lastAnimationTime<this.animationCooldown){console.log("Animation skipped - already running or too soon since last animation");return}this.isAnimating=!0,this.lastAnimationTime=i,this.lastStats=a,console.log("Starting stat animations...",{userStats:t,currentRoom:o,isConnected:s});const r=o&&t.roomScores[o]||0,c=s?"connected":"disconnected",l=o||"No Room";this.getRandomGreeting(n),e.innerHTML=`
            <div class="stat-item room-status ${c}">
                <span class="stat-label">Room:</span>
                <span class="stat-value" id="room-name-display">${l}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Score:</span>
                <span class="stat-value" id="total-score-display">0</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Room Score:</span>
                <span class="stat-value" id="room-score-display">0</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Messages:</span>
                <span class="stat-value" id="messages-display">0</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Comments:</span>
                <span class="stat-value" id="comments-display">0</span>
            </div>
        `,console.log("Starting count-up animations..."),this.startCountUpAnimations(e,t,r),this.startRoomNameAnimation(e,l),setTimeout(()=>{this.isAnimating=!1,console.log("Animations completed")},Math.max(this.animationDuration,this.roomAnimationDuration)+500)}startCountUpAnimations(e,t,o){this.animateCountUp(e,"total-score-display",t.totalScore,this.animationDuration),this.animateCountUp(e,"room-score-display",o,this.animationDuration),this.animateCountUp(e,"messages-display",t.totalMessages,this.animationDuration),this.animateCountUp(e,"comments-display",t.totalComments,this.animationDuration)}animateCountUp(e,t,o,s){const n=e.querySelector(`#${t}`);if(!n){console.warn(`Element with id ${t} not found in container`);return}console.log(`Animating ${t} from 0 to ${o}`);const i=performance.now(),a=0,r=c=>{const l=c-i,d=Math.min(l/s,1),h=1-Math.pow(1-d,3),m=Math.floor(a+(o-a)*h);n.textContent=m,d<1?requestAnimationFrame(r):(n.textContent=o,console.log(`Animation completed for ${t}: ${o}`))};requestAnimationFrame(r)}startRoomNameAnimation(e,t){const o=e.querySelector("#room-name-display");if(!o){console.warn("Room name display element not found in container");return}console.log(`Starting room name animation for: "${t}"`);const s=t.length;let n=Array(s).fill("").map(()=>this.getRandomSymbol()),i=new Set,a=performance.now();const r=c=>{if((c-a)/this.roomAnimationDuration<1){if(Math.random()<.3&&i.size<s){const m=Math.floor(Math.random()*s);i.has(m)||i.add(m)}const h=n.map((m,u)=>i.has(u)?t[u]:m);o.textContent=h.join(""),n=n.map((m,u)=>i.has(u)?m:this.getRandomSymbol()),setTimeout(()=>{requestAnimationFrame(r)},this.roomSymbolInterval)}else o.textContent=t,console.log(`Room name animation completed: "${t}"`)};requestAnimationFrame(r)}getRandomSymbol(){return this.symbols[Math.floor(Math.random()*this.symbols.length)]}pulseStat(e,t){const o=e.querySelector(`#${t}`);o&&(o.style.transition="transform 0.3s ease, color 0.3s ease",o.style.transform="scale(1.2)",o.style.color="#ff6b6b",setTimeout(()=>{o.style.transform="scale(1)",o.style.color=""},300))}addStatsGlow(){const e=document.getElementById("user-stats-content");e&&(e.style.boxShadow="0 0 20px rgba(102, 126, 234, 0.5)",e.style.transition="box-shadow 0.5s ease",setTimeout(()=>{e.style.boxShadow=""},2e3))}animateStatUpdate(e,t,o){const s=`${t}-display`,n=e.querySelector(`#${s}`);n&&(parseInt(n.textContent),this.pulseStat(e,s),setTimeout(()=>{this.animateCountUp(e,s,o,500)},100))}resetAnimations(){this.isAnimating=!1,this.lastAnimationTime=0,this.lastStats=null,console.log("Animations reset")}forceNewGreeting(e){const t=this.greetingMessages[Math.floor(Math.random()*this.greetingMessages.length)];return this.sessionGreeting=t.replace("{name}",e||"there"),console.log(`Forced new greeting: "${this.sessionGreeting}"`),this.sessionGreeting}forceAnimateStatsDisplay(e,t,o,s,n=null){this.resetAnimations(),this.animateStatsDisplay(e,t,o,s,n)}}class B{constructor(e={}){this.onCommentSave=e.onCommentSave||(()=>{}),this.onCommentDelete=e.onCommentDelete||(()=>{}),this.onCommentResolve=e.onCommentResolve||(()=>{}),this.isInitialized=!1,this.activeTooltip=null,console.log("📝 CommentUI initialized")}initialize(){try{this.setupTextSelection(),this.setupKeyboardShortcuts(),this.isInitialized=!0,console.log("📝 CommentUI fully initialized")}catch(e){throw console.error("❌ Failed to initialize CommentUI:",e),e}}setupTextSelection(){document.addEventListener("mouseup",e=>{const o=window.getSelection().toString().trim();o.length>0&&setTimeout(()=>{const s=window.getSelection();s.toString().trim()===o&&this.handleTextSelection(e,s)},100)}),document.addEventListener("dblclick",e=>{const t=window.getSelection();t.toString().trim().length>0&&this.handleTextSelection(e,t)})}handleTextSelection(e,t){try{if(t.toString().trim().length<3)return;const s=this.extractSelectionData(t);if(!s)return;this.showCommentPanel(s)}catch(o){console.error("❌ Failed to handle text selection:",o)}}extractSelectionData(e){try{if(e.rangeCount===0)return null;const t=e.getRangeAt(0),o=t.getBoundingClientRect(),s=e.toString().trim(),n=t.toString(),i=t.startOffset,a=t.endOffset,r=t.commonAncestorContainer.parentElement||t.commonAncestorContainer,c=r.tagName?.toLowerCase()||"unknown",l=r.textContent||"",d=l.indexOf(s),h=d!==-1?d:i,m=d!==-1?d+s.length:a,u=l.substring(0,h),b=l.substring(m),f=l.substring(Math.max(0,h-50),h),v=l.substring(m,Math.min(l.length,m+50)),re=this.getElementPath(r);return console.log("📝 Selection data extracted:",{selectedText:s,rangeText:n,parentText:l.substring(0,100)+"...",textIndex:d,startOffset:h,endOffset:m}),{text:s,selectedText:s,url:window.location.href,title:document.title,startOffset:h,endOffset:m,parentTag:c,parentText:l,parentTextBefore:u,parentTextAfter:b,contextBefore:f,contextAfter:v,elementPath:re,rect:{top:o.top,left:o.left,width:o.width,height:o.height},timestamp:Date.now()}}catch(t){return console.error("❌ Failed to extract selection data:",t),null}}getElementPath(e){const t=[];let o=e;for(;o&&o!==document.body;){let s=o.tagName.toLowerCase();if(o.id)s+=`#${o.id}`;else if(o.className){const n=o.className.split(" ").filter(i=>i.trim()).slice(0,2);n.length>0&&(s+=`.${n.join(".")}`)}t.unshift(s),o=o.parentElement}return t.join(" > ")}async showCommentPanel(e){try{this.removeActiveTooltip();const t=this.createCommentTooltip(e);document.body.appendChild(t),this.activeTooltip=t,this.positionTooltip(t,e.rect);const o=t.querySelector("textarea");o&&o.focus()}catch(t){console.error("❌ Failed to show comment panel:",t)}}createCommentTooltip(e){const t=document.createElement("div");return t.className="comment-tooltip",t.innerHTML=`
            <div class="comment-tooltip-header">
                <span class="comment-tooltip-title">Add Comment</span>
                <button class="comment-tooltip-close" type="button">×</button>
            </div>
            <div class="comment-tooltip-content">
                <div class="comment-tooltip-selection">
                    <strong>Selected text:</strong>
                    <div class="comment-tooltip-text">"${this.escapeHtml(e.text)}"</div>
                </div>
                <textarea 
                    class="comment-tooltip-input" 
                    placeholder="Add your comment..."
                    rows="3"
                ></textarea>
                <div class="comment-tooltip-actions">
                    <button class="comment-tooltip-cancel" type="button">Cancel</button>
                    <button class="comment-tooltip-save" type="button">Save Comment</button>
                </div>
            </div>
        `,this.setupTooltipEvents(t,e),t}setupTooltipEvents(e,t){const o=e.querySelector(".comment-tooltip-close"),s=e.querySelector(".comment-tooltip-cancel"),n=e.querySelector(".comment-tooltip-save"),i=e.querySelector(".comment-tooltip-input");o.addEventListener("click",()=>{this.removeActiveTooltip()}),s.addEventListener("click",()=>{this.removeActiveTooltip()}),n.addEventListener("click",()=>{const a=i.value.trim();a&&this.onCommentSave(a,t),this.removeActiveTooltip()}),i.addEventListener("keydown",a=>{if(a.ctrlKey&&a.key==="Enter"){const r=i.value.trim();r&&(this.onCommentSave(r,t),this.removeActiveTooltip())}else a.key==="Escape"&&this.removeActiveTooltip()}),document.addEventListener("click",a=>{e.contains(a.target)||this.removeActiveTooltip()})}positionTooltip(e,t){const o=e.getBoundingClientRect(),s=window.innerWidth,n=window.innerHeight;let i=t.top+t.height+10,a=t.left;a+o.width>s&&(a=s-o.width-10),i+o.height>n&&(i=t.top-o.height-10),a<10&&(a=10),e.style.position="absolute",e.style.top=`${i}px`,e.style.left=`${a}px`,e.style.zIndex="10000"}removeActiveTooltip(){this.activeTooltip&&(this.activeTooltip.remove(),this.activeTooltip=null)}setupKeyboardShortcuts(){document.addEventListener("keydown",e=>{if(e.ctrlKey&&e.shiftKey&&e.key==="C"){e.preventDefault();const t=window.getSelection();if(t.toString().trim().length>=3){const s=this.extractSelectionData(t);s&&this.showCommentPanel(s)}}})}addHighlightHoverTooltip(e,t){try{let o=null;const s=i=>{if(o)return;o=document.createElement("div"),o.className="comment-hover-tooltip",o.innerHTML=`
                    <div class="comment-hover-header">
                        <span class="comment-hover-author">${t.author}</span>
                        <span class="comment-hover-time">${this.formatTimestamp(t.timestamp)}</span>
                    </div>
                    <div class="comment-hover-content">${this.escapeHtml(t.text)}</div>
                    ${t.resolved?'<div class="comment-hover-resolved">✓ Resolved</div>':""}
                `,document.body.appendChild(o);const a=e.getBoundingClientRect();o.style.position="absolute",o.style.top=`${a.top-o.offsetHeight-5}px`,o.style.left=`${a.left}px`,o.style.zIndex="10001"},n=()=>{o&&(o.remove(),o=null)};e.addEventListener("mouseenter",s),e.addEventListener("mouseleave",n)}catch(o){console.error("❌ Failed to add hover tooltip:",o)}}formatTimestamp(e){const t=new Date(e),s=new Date-t,n=Math.floor(s/6e4),i=Math.floor(s/36e5),a=Math.floor(s/864e5);return n<1?"Just now":n<60?`${n}m ago`:i<24?`${i}h ago`:a<7?`${a}d ago`:t.toLocaleDateString()}escapeHtml(e){const t=document.createElement("div");return t.textContent=e,t.innerHTML}destroy(){try{this.removeActiveTooltip(),this.isInitialized=!1,console.log("📝 CommentUI destroyed")}catch(e){console.error("❌ Failed to destroy CommentUI:",e)}}}class j{constructor(e={}){this.currentRoom=e.currentRoom||"default",this.dbName="commentDB",this.dbVersion=1,this.db=null,console.log("📝 CommentData initialized")}async initialize(){try{this.db=await this.openDatabase(),console.log("📝 CommentData database initialized")}catch(e){throw console.error("❌ Failed to initialize CommentData:",e),e}}openDatabase(){return new Promise((e,t)=>{const o=indexedDB.open(this.dbName,this.dbVersion);o.onerror=()=>{t(new Error("Failed to open comment database"))},o.onsuccess=()=>{e(o.result)},o.onupgradeneeded=s=>{const n=s.target.result;if(!n.objectStoreNames.contains("comments")){const i=n.createObjectStore("comments",{keyPath:"id"});i.createIndex("url","url",{unique:!1}),i.createIndex("room","room",{unique:!1}),i.createIndex("highlightId","highlightId",{unique:!1}),i.createIndex("peerId","peerId",{unique:!1}),i.createIndex("timestamp","timestamp",{unique:!1}),i.createIndex("resolved","resolved",{unique:!1})}}})}updateCurrentRoom(e){this.currentRoom=e}async saveComment(e){try{if(!this.db)throw new Error("Database not initialized");const o=this.db.transaction(["comments"],"readwrite").objectStore("comments");return new Promise((s,n)=>{const i=o.put(e);i.onsuccess=()=>{console.log("📝 Comment saved to database:",e.id),s(e)},i.onerror=()=>{n(new Error("Failed to save comment"))}})}catch(t){throw console.error("❌ Failed to save comment:",t),t}}async getCommentById(e){try{if(!this.db)throw new Error("Database not initialized");const o=this.db.transaction(["comments"],"readonly").objectStore("comments");return new Promise((s,n)=>{const i=o.get(e);i.onsuccess=()=>{s(i.result)},i.onerror=()=>{n(new Error("Failed to get comment"))}})}catch(t){throw console.error("❌ Failed to get comment:",t),t}}async getCommentsByUrlAndRoom(e,t){try{if(!this.db)throw new Error("Database not initialized");const n=this.db.transaction(["comments"],"readonly").objectStore("comments").index("url");return new Promise((i,a)=>{const r=n.getAll(e);r.onsuccess=()=>{const c=r.result.filter(l=>l.room===t);i(c)},r.onerror=()=>{a(new Error("Failed to get comments by URL"))}})}catch(o){throw console.error("❌ Failed to get comments by URL:",o),o}}async getCommentsByHighlightId(e){try{if(!this.db)throw new Error("Database not initialized");const s=this.db.transaction(["comments"],"readonly").objectStore("comments").index("highlightId");return new Promise((n,i)=>{const a=s.getAll(e);a.onsuccess=()=>{n(a.result)},a.onerror=()=>{i(new Error("Failed to get comments by highlight ID"))}})}catch(t){throw console.error("❌ Failed to get comments by highlight ID:",t),t}}async getCommentsByPeerId(e){try{if(!this.db)throw new Error("Database not initialized");const s=this.db.transaction(["comments"],"readonly").objectStore("comments").index("peerId");return new Promise((n,i)=>{const a=s.getAll(e);a.onsuccess=()=>{n(a.result)},a.onerror=()=>{i(new Error("Failed to get comments by peer ID"))}})}catch(t){throw console.error("❌ Failed to get comments by peer ID:",t),t}}async getUnresolvedComments(e,t){try{return(await this.getCommentsByUrlAndRoom(e,t)).filter(s=>!s.resolved)}catch(o){throw console.error("❌ Failed to get unresolved comments:",o),o}}async deleteComment(e){try{if(!this.db)throw new Error("Database not initialized");const o=this.db.transaction(["comments"],"readwrite").objectStore("comments");return new Promise((s,n)=>{const i=o.delete(e);i.onsuccess=()=>{console.log("📝 Comment deleted from database:",e),s()},i.onerror=()=>{n(new Error("Failed to delete comment"))}})}catch(t){throw console.error("❌ Failed to delete comment:",t),t}}async resolveComment(e,t){try{const o=await this.getCommentById(e);if(!o)throw new Error("Comment not found");o.resolved=!0,o.resolvedBy=t,o.resolvedAt=Date.now(),await this.saveComment(o),console.log("📝 Comment resolved:",e)}catch(o){throw console.error("❌ Failed to resolve comment:",o),o}}async getCommentStats(e,t){try{const o=await this.getCommentsByUrlAndRoom(e,t),s=o.filter(i=>i.resolved).length,n=o.length-s;return{total:o.length,resolved:s,unresolved:n,byPeer:this.groupCommentsByPeer(o)}}catch(o){return console.error("❌ Failed to get comment stats:",o),{total:0,resolved:0,unresolved:0,byPeer:{}}}}groupCommentsByPeer(e){const t={};for(const o of e){const s=o.peerId;t[s]||(t[s]={total:0,resolved:0,unresolved:0}),t[s].total++,o.resolved?t[s].resolved++:t[s].unresolved++}return t}async clearCommentsForRoom(e){try{if(!this.db)throw new Error("Database not initialized");const o=this.db.transaction(["comments"],"readwrite").objectStore("comments"),s=o.index("room");return new Promise((n,i)=>{const a=s.getAll(e);a.onsuccess=()=>{const r=a.result,c=r.map(l=>new Promise((d,h)=>{const m=o.delete(l.id);m.onsuccess=()=>d(),m.onerror=()=>h()}));Promise.all(c).then(()=>{console.log(`📝 Cleared ${r.length} comments for room: ${e}`),n()}).catch(i)},a.onerror=()=>{i(new Error("Failed to clear comments for room"))}})}catch(t){throw console.error("❌ Failed to clear comments for room:",t),t}}async exportComments(e,t){try{const o=await this.getCommentsByUrlAndRoom(e,t);return{url:e,room:t,exportedAt:Date.now(),comments:o}}catch(o){throw console.error("❌ Failed to export comments:",o),o}}async importComments(e){try{const t=e.comments.map(o=>(o.room=e.room,this.saveComment(o)));await Promise.all(t),console.log(`📝 Imported ${e.comments.length} comments`)}catch(t){throw console.error("❌ Failed to import comments:",t),t}}destroy(){try{this.db&&(this.db.close(),this.db=null),console.log("📝 CommentData destroyed")}catch(e){console.error("❌ Failed to destroy CommentData:",e)}}}class Q{constructor(e={}){this.webrtcChat=e.webrtcChat||null,this.onCommentAdd=e.onCommentAdd||(()=>{}),this.onCommentDelete=e.onCommentDelete||(()=>{}),this.onCommentResolve=e.onCommentResolve||(()=>{}),this.onCommentStatusRequest=e.onCommentStatusRequest||(()=>{}),this.onCommentStatusResponse=e.onCommentStatusResponse||(()=>{}),console.log("📝 CommentSync initialized")}updateWebRTCChat(e){this.webrtcChat=e}async broadcastCommentAdd(e){try{if(!this.webrtcChat){console.warn("⚠️ WebRTC chat not available for comment broadcast");return}const t={type:"comment_add",data:e};this.webrtcChat.broadcastMessage(t),console.log("📝 Comment add broadcasted:",e.id)}catch(t){console.error("❌ Failed to broadcast comment add:",t)}}async broadcastCommentDelete(e){try{if(!this.webrtcChat){console.warn("⚠️ WebRTC chat not available for comment broadcast");return}const t={type:"comment_delete",data:{commentId:e}};this.webrtcChat.broadcastMessage(t),console.log("📝 Comment delete broadcasted:",e)}catch(t){console.error("❌ Failed to broadcast comment delete:",t)}}async broadcastCommentResolve(e,t,o,s,n){try{if(!this.webrtcChat){console.warn("⚠️ WebRTC chat not available for comment broadcast");return}const i={type:"comment_resolve",data:{commentId:e,highlightId:t,url:o,room:s,resolvedBy:n}};this.webrtcChat.broadcastMessage(i),console.log("📝 Comment resolve broadcasted:",e)}catch(i){console.error("❌ Failed to broadcast comment resolve:",i)}}async broadcastCommentStatusRequest(e){try{if(!this.webrtcChat){console.warn("⚠️ WebRTC chat not available for comment broadcast");return}const t={type:"comment_status_request",data:{commentIds:e,url:window.location.href,room:this.webrtcChat.roomName||"default"}};this.webrtcChat.broadcastMessage(t),console.log("📝 Comment status request broadcasted:",e.length,"comments")}catch(t){console.error("❌ Failed to broadcast comment status request:",t)}}async broadcastCommentStatusResponse(e,t,o){try{if(!this.webrtcChat){console.warn("⚠️ WebRTC chat not available for comment broadcast");return}const s={type:"comment_status_response",data:{resolvedCommentIds:e,url:t,room:o}};this.webrtcChat.broadcastMessage(s),console.log("📝 Comment status response broadcasted:",e.length,"resolved comments")}catch(s){console.error("❌ Failed to broadcast comment status response:",s)}}async broadcastSystemMessage(e){try{if(!this.webrtcChat){console.warn("⚠️ WebRTC chat not available for system message");return}const t=this.webrtcChat.myUsername||"You",o=document.title||"this page",s={message:`💬 ${t} added a comment on "${o}"`,url:window.location.href,commentId:e.id,author:t,pageTitle:o},n={type:"system_message",payload:s};this.webrtcChat.broadcastMessage(n),this.webrtcChat.onSystemMessage&&this.webrtcChat.onSystemMessage(s,this.webrtcChat.myClientId),console.log("📝 System message broadcasted for comment:",e.id)}catch(t){console.error("❌ Failed to broadcast system message:",t)}}handleCommentAdd(e,t){try{console.log("📝 Received comment add from peer:",t,e.id),this.onCommentAdd(e,t)}catch(o){console.error("❌ Failed to handle comment add:",o)}}handleCommentDelete(e,t){try{console.log("📝 Received comment delete from peer:",t,e.commentId),this.onCommentDelete(e.commentId,t)}catch(o){console.error("❌ Failed to handle comment delete:",o)}}handleCommentResolve(e,t){try{console.log("📝 Received comment resolve from peer:",t,e.commentId),this.onCommentResolve(e.commentId,e.highlightId,e.url,e.room,e.resolvedBy,t)}catch(o){console.error("❌ Failed to handle comment resolve:",o)}}handleCommentStatusRequest(e,t){try{console.log("📝 Received comment status request from peer:",t),this.onCommentStatusRequest(e,t)}catch(o){console.error("❌ Failed to handle comment status request:",o)}}handleCommentStatusResponse(e,t){try{console.log("📝 Received comment status response from peer:",t),this.onCommentStatusResponse(e,t)}catch(o){console.error("❌ Failed to handle comment status response:",o)}}async requestCommentStatusFromPeers(e){try{if(!this.webrtcChat||Object.keys(this.webrtcChat.peers).length===0){console.log("📝 No peers available for comment status request");return}await this.broadcastCommentStatusRequest(e),console.log("📝 Comment status requested from all peers")}catch(t){console.error("❌ Failed to request comment status from peers:",t)}}async syncCommentsOnConnection(){try{if(!this.webrtcChat){console.warn("⚠️ WebRTC chat not available for comment sync");return}const e=await this.getUnresolvedCommentIds();e.length>0&&(await this.requestCommentStatusFromPeers(e),console.log("📝 Comment sync initiated with",e.length,"unresolved comments"))}catch(e){console.error("❌ Failed to sync comments on connection:",e)}}async getUnresolvedCommentIds(){try{return[]}catch(e){return console.error("❌ Failed to get unresolved comment IDs:",e),[]}}onPeerConnected(e){try{console.log("📝 Peer connected, initiating comment sync:",e),this.syncCommentsOnConnection()}catch(t){console.error("❌ Failed to handle peer connection:",t)}}onPeerDisconnected(e){try{console.log("📝 Peer disconnected:",e)}catch(t){console.error("❌ Failed to handle peer disconnection:",t)}}getConnectionStatus(){return this.webrtcChat?{connected:this.webrtcChat.isConnected(),peerCount:Object.keys(this.webrtcChat.peers).length,isHub:this.webrtcChat.isHubUser()}:{connected:!1,peerCount:0}}destroy(){try{this.webrtcChat=null,console.log("📝 CommentSync destroyed")}catch(e){console.error("❌ Failed to destroy CommentSync:",e)}}}class _{constructor(e={}){this.currentRoom=e.currentRoom||"default",this.highlights=new Map,this.commentHighlights=new Map,console.log("📝 CommentHighlights initialized")}initialize(){try{this.setupHighlightStyles(),this.removePlaceholderHighlights(),console.log("📝 CommentHighlights fully initialized")}catch(e){throw console.error("❌ Failed to initialize CommentHighlights:",e),e}}updateCurrentRoom(e){this.currentRoom=e}setupHighlightStyles(){if(document.getElementById("comment-highlight-styles"))return;const e=document.createElement("style");e.id="comment-highlight-styles",e.textContent=`
            .comment-highlight {
                background-color: rgba(255, 255, 0, 0.3);
                border-bottom: 2px solid #ffd700;
                cursor: pointer;
                transition: background-color 0.2s ease;
            }
            
            .comment-highlight:hover {
                background-color: rgba(255, 255, 0, 0.5);
            }
            
            .comment-highlight.resolved {
                background-color: rgba(0, 255, 0, 0.3);
                border-bottom-color: #00ff00;
            }
            
            .comment-highlight.resolved:hover {
                background-color: rgba(0, 255, 0, 0.5);
            }
            
            .comment-tooltip {
                background: white;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
                max-width: 400px;
                min-width: 300px;
                z-index: 10000;
            }
            
            .comment-tooltip-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 16px;
                border-bottom: 1px solid #eee;
                background: #f8f9fa;
                border-radius: 8px 8px 0 0;
            }
            
            .comment-tooltip-title {
                font-weight: 600;
                color: #333;
            }
            
            .comment-tooltip-close {
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
                color: #666;
                padding: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .comment-tooltip-close:hover {
                color: #333;
            }
            
            .comment-tooltip-content {
                padding: 16px;
            }
            
            .comment-tooltip-selection {
                margin-bottom: 12px;
            }
            
            .comment-tooltip-selection strong {
                display: block;
                margin-bottom: 4px;
                color: #555;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .comment-tooltip-text {
                background: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
                font-style: italic;
                color: #666;
                border-left: 3px solid #ffd700;
                max-height: 60px;
                overflow-y: auto;
            }
            
            .comment-tooltip-input {
                width: 100%;
                min-height: 60px;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: inherit;
                font-size: 14px;
                resize: vertical;
                box-sizing: border-box;
            }
            
            .comment-tooltip-input:focus {
                outline: none;
                border-color: #007bff;
                box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
            }
            
            .comment-tooltip-actions {
                display: flex;
                justify-content: flex-end;
                gap: 8px;
                margin-top: 12px;
            }
            
            .comment-tooltip-cancel,
            .comment-tooltip-save {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                cursor: pointer;
                transition: background-color 0.2s ease;
            }
            
            .comment-tooltip-cancel {
                background: #f8f9fa;
                color: #666;
            }
            
            .comment-tooltip-cancel:hover {
                background: #e9ecef;
            }
            
            .comment-tooltip-save {
                background: #007bff;
                color: white;
            }
            
            .comment-tooltip-save:hover {
                background: #0056b3;
            }
            
            .comment-hover-tooltip {
                background: white;
                border: 1px solid #ccc;
                border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 13px;
                max-width: 300px;
                z-index: 10001;
                transition: opacity 0.2s ease;
                position: relative;
            }
            
            .comment-hover-tooltip::after {
                content: '';
                position: absolute;
                bottom: -6px;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid white;
            }
            
            .comment-hover-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 12px;
                border-bottom: 1px solid #eee;
                background: #f8f9fa;
                border-radius: 6px 6px 0 0;
            }
            
            .comment-hover-author {
                font-weight: 600;
                color: #333;
            }
            
            .comment-hover-time {
                font-size: 11px;
                color: #666;
            }
            
            .comment-hover-content {
                padding: 12px;
                color: #555;
                line-height: 1.4;
            }
            
            .comment-hover-resolved {
                padding: 4px 12px;
                background: #d4edda;
                color: #155724;
                font-size: 11px;
                font-weight: 600;
                text-align: center;
            }
            
            .comment-hover-actions {
                padding: 8px 12px;
                border-top: 1px solid #eee;
                background: #f8f9fa;
                border-radius: 0 0 6px 6px;
            }
            
            .comment-resolve-btn {
                background: #28a745;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: background-color 0.2s ease;
                width: 100%;
                display: block;
            }
            
            .comment-resolve-btn:hover {
                background: #218838;
            }
            
            .comment-resolve-btn:active {
                background: #1e7e34;
            }
            
            .comment-author-link {
                color: #3b82f6;
                text-decoration: none;
                font-weight: 500;
                transition: color 0.2s ease;
            }
            
            .comment-author-link:hover {
                color: #1d4ed8;
                text-decoration: underline;
            }
            
            .comment-highlight-placeholder {
                position: fixed !important;
                top: 20px !important;
                right: 20px !important;
                background: rgba(255, 255, 0, 0.3) !important;
                border: 2px solid #ffd700 !important;
                padding: 8px 12px !important;
                border-radius: 6px !important;
                font-size: 14px !important;
                z-index: 10000 !important;
                max-width: 300px !important;
                word-wrap: break-word !important;
                cursor: pointer !important;
                transition: all 0.2s ease !important;
            }
            
            .comment-highlight-placeholder:hover {
                background: rgba(255, 255, 0, 0.5) !important;
                transform: scale(1.05) !important;
            }
            
            .comment-highlight-placeholder.resolved {
                background: rgba(0, 255, 0, 0.3) !important;
                border-color: #00ff00 !important;
            }
        `,document.head.appendChild(e)}async createHighlightFromComment(e){try{if(!e.highlightData){console.warn("⚠️ No highlight data for comment:",e.id);return}if(e.resolved){console.log("📝 Skipping highlight creation for resolved comment:",e.id);return}if(this.commentHighlights.has(e.id)){console.log("📝 Highlight already exists for comment:",e.id);return}const t=await this.createHighlight(e.highlightData,e);t&&(this.highlights.set(e.highlightId,t),this.commentHighlights.set(e.id,e.highlightId),this.addHoverTooltip(t,e),console.log("📝 Highlight created for comment:",e.id))}catch(t){console.error("❌ Failed to create highlight from comment:",t)}}async createHighlight(e,t){try{console.log("📝 Creating highlight with data:",{text:e.text?.substring(0,50)+"...",textLength:e.text?.length,contextBefore:e.contextBefore?.substring(0,30)+"...",contextAfter:e.contextAfter?.substring(0,30)+"...",parentTag:e.parentTag,startOffset:e.startOffset,endOffset:e.endOffset});let o=this.findTextNode(e),s=null;return o?(console.log("📝 Found text node, creating highlight"),s=await this.createHighlightFromTextNode(o,e,t)):console.log("📝 No text node found, trying content search"),s||(console.log("📝 Trying content-based highlight creation"),s=await this.createHighlightByContent(e,t)),s?console.log("📝 Highlight created successfully"):(console.warn("⚠️ All highlight methods failed, creating placeholder highlight"),s=await this.createPlaceholderHighlight(e,t)),s}catch(o){return console.error("❌ Failed to create highlight element:",o),null}}async createHighlightFromTextNode(e,t,o){try{const s=document.createElement("span");s.className="comment-highlight",s.dataset.highlightId=o.highlightId,s.dataset.commentId=o.id,console.log("📝 Comment resolved status:",{commentId:o.id,resolved:o.resolved,resolvedType:typeof o.resolved}),o.resolved&&(s.classList.add("resolved"),console.log("📝 Added resolved class to highlight for comment:",o.id));const n=e.textContent,i=t.text,a=n.indexOf(i);if(a===-1)return console.log("📝 Selected text not found in this text node"),null;const r=document.createRange();r.setStart(e,a),r.setEnd(e,a+i.length);try{r.surroundContents(s)}catch{const l=r.extractContents();s.appendChild(l),r.insertNode(s)}return s}catch(s){return console.error("❌ Failed to create highlight from text node:",s),null}}async createHighlightByContent(e,t){try{const o=e.text||e.selectedText;if(!o||o.length<1)return console.log("📝 No selected text for content search"),null;console.log("📝 Searching for text content:",o.substring(0,50)+"...");const s=this.getAllTextNodes();let n=null,i=0;for(const a of s){const r=a.textContent;if(r.trim().length<o.length)continue;const c=r.indexOf(o);if(c!==-1){let l=1;if(e.contextBefore){const d=r.substring(Math.max(0,c-50),c),h=e.contextBefore.trim();h.length>0&&d.includes(h.slice(-Math.min(20,h.length)))&&(l+=2)}if(e.contextAfter){const d=r.substring(c+o.length,c+o.length+50),h=e.contextAfter.trim();h.length>0&&d.includes(h.slice(0,Math.min(20,h.length)))&&(l+=2)}if(e.parentTag){const d=a.parentElement;d&&d.tagName.toLowerCase()===e.parentTag.toLowerCase()&&(l+=1)}e.startOffset!==void 0&&e.endOffset!==void 0&&(e.parentText||"").indexOf(o)===c&&(l+=3),l>i&&(i=l,n={textNode:a,index:c,score:l},console.log("📝 Found better content match with score:",l))}}if(n&&i>0){console.log("📝 Found best text match with score:",i);const a=document.createElement("span");a.className="comment-highlight",a.dataset.highlightId=t.highlightId,a.dataset.commentId=t.id,console.log("📝 Comment resolved status (content search):",{commentId:t.id,resolved:t.resolved,resolvedType:typeof t.resolved}),t.resolved&&(a.classList.add("resolved"),console.log("📝 Added resolved class to highlight for comment (content search):",t.id));const r=document.createRange();r.setStart(n.textNode,n.index),r.setEnd(n.textNode,n.index+o.length);try{return r.surroundContents(a),console.log("📝 Successfully wrapped text with highlight"),a}catch(c){console.warn("⚠️ Could not wrap found text:",c);try{const l=r.extractContents();return a.appendChild(l),r.insertNode(a),console.log("📝 Successfully wrapped text with alternative method"),a}catch(l){console.warn("⚠️ Alternative wrapping also failed:",l)}}}if(!n&&o.length>10){console.log("📝 Trying aggressive partial matching...");for(const a of s){const r=a.textContent;if(!(r.trim().length<10))for(let c=Math.min(o.length,50);c>=10;c-=5){const l=o.substring(0,c);if(r.includes(l)){const d=r.indexOf(l);console.log("📝 Found aggressive partial match:",l.substring(0,30)+"...");const h=document.createElement("span");h.className="comment-highlight",h.dataset.highlightId=t.highlightId,h.dataset.commentId=t.id,console.log("📝 Comment resolved status (placeholder):",{commentId:t.id,resolved:t.resolved,resolvedType:typeof t.resolved}),t.resolved&&(h.classList.add("resolved"),console.log("📝 Added resolved class to highlight for comment (placeholder):",t.id));const m=document.createRange();m.setStart(a,d),m.setEnd(a,d+l.length);try{return m.surroundContents(h),console.log("📝 Successfully wrapped partial text with highlight"),h}catch(u){console.warn("⚠️ Could not wrap partial text:",u);try{const b=m.extractContents();return h.appendChild(b),m.insertNode(h),console.log("📝 Successfully wrapped partial text with alternative method"),h}catch(b){console.warn("⚠️ Alternative partial wrapping also failed:",b)}}}}}}return console.log("📝 No suitable content match found"),null}catch(o){return console.error("❌ Failed to create highlight by content:",o),null}}async createPlaceholderHighlight(e,t){try{return console.warn("⚠️ Placeholder highlights are disabled - highlight creation failed"),console.log("📝 Highlight data that failed:",{text:e.text?.substring(0,50)+"...",textLength:e.text?.length,contextBefore:e.contextBefore?.substring(0,30)+"...",contextAfter:e.contextAfter?.substring(0,30)+"...",parentTag:e.parentTag}),null}catch(o){return console.error("❌ Failed to create placeholder highlight:",o),null}}removePlaceholderHighlights(){try{const e=document.querySelectorAll(".comment-highlight-placeholder");e.forEach(t=>{console.log("📝 Removing placeholder highlight:",t.dataset.highlightId),t.remove()}),e.length>0&&console.log(`📝 Removed ${e.length} placeholder highlights`)}catch(e){console.error("❌ Failed to remove placeholder highlights:",e)}}getAllTextNodes(){const e=[],t=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT,null,!1);let o;for(;o=t.nextNode();)o.textContent.trim().length>0&&e.push(o);return e}findTextNode(e){try{const t=e.text||e.selectedText;if(!t||t.length<1)return console.log("📝 No selected text provided"),null;console.log("📝 Searching for text node with selected text:",t.substring(0,50)+"...");const o=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT,null,!1);let s=null,n=0,i;for(;i=o.nextNode();){const a=i.textContent;if(!(a.trim().length<t.length)&&a.includes(t)){let r=1;const c=a.indexOf(t);if(e.startOffset!==void 0&&e.endOffset!==void 0&&(e.parentText||"").indexOf(t)===c&&(r+=3),e.contextBefore&&e.contextAfter){const l=e.contextBefore.trim(),d=e.contextAfter.trim();l.length>0&&a.substring(Math.max(0,c-50),c).includes(l.slice(-Math.min(20,l.length)))&&(r+=2),d.length>0&&a.substring(c+t.length,c+t.length+50).includes(d.slice(0,Math.min(20,d.length)))&&(r+=2)}if(e.parentTag){const l=i.parentElement;l&&l.tagName.toLowerCase()===e.parentTag.toLowerCase()&&(r+=1)}r>n&&(n=r,s=i,console.log("📝 Found better text node match with score:",r,"at index:",c))}}if(!s&&t.length>20)for(console.log("📝 No exact match found, trying partial matching..."),o.currentNode=document.body;i=o.nextNode();){const a=i.textContent;if(a.trim().length<t.length)continue;const r=Math.min(t.length,30),c=t.substring(0,r);if(a.includes(c)){let l=.5;const d=a.indexOf(c);d>0&&/\s/.test(a[d-1])&&(l+=.5),l>n&&(n=l,s=i,console.log("📝 Found partial text node match with score:",l))}}return s&&n>0?(console.log("📝 Found best text node match with score:",n),s):(console.log("📝 No suitable text node found"),null)}catch(t){return console.error("❌ Failed to find text node:",t),null}}addHoverTooltip(e,t){try{let o=null,s=null,n=null;const i=r=>{n&&(clearTimeout(n),n=null),!o&&(s=setTimeout(()=>{if(o)return;o=document.createElement("div"),o.className="comment-hover-tooltip",o.innerHTML=`
                        <div class="comment-hover-header">
                            <span class="comment-hover-author">${this.createClickableAuthor(t)}</span>
                            <span class="comment-hover-time">${this.formatTimestamp(t.timestamp)}</span>
                        </div>
                        <div class="comment-hover-content">${this.escapeHtml(t.text)}</div>
                        ${t.resolved?'<div class="comment-hover-resolved">✓ Resolved</div>':""}
                        ${t.resolved?"":`
                            <div class="comment-hover-actions">
                                <button class="comment-resolve-btn" data-comment-id="${t.id}" data-highlight-id="${t.highlightId}">
                                    ✓ Resolve
                                </button>
                            </div>
                        `}
                    `,document.body.appendChild(o),o.addEventListener("mouseenter",()=>{n&&(clearTimeout(n),n=null)}),o.addEventListener("mouseleave",()=>{a()});const c=o.querySelector(".comment-resolve-btn");c&&c.addEventListener("click",l=>{l.preventDefault(),l.stopPropagation();const d=c.dataset.commentId,h=c.dataset.highlightId;this.resolveComment(d,h),a()}),setTimeout(()=>{if(o){const l=e.getBoundingClientRect(),d=o.getBoundingClientRect();let h=l.top+window.scrollY,m=l.left+window.scrollX;m=m+l.width/2-d.width/2,h=l.top+window.scrollY-d.height,h<window.scrollY+10&&(h=l.bottom+window.scrollY),m+d.width>window.innerWidth-10&&(m=window.innerWidth-d.width-10),m<10&&(m=10),h+d.height>window.innerHeight+window.scrollY-10&&(h=window.innerHeight+window.scrollY-d.height-10),o.style.position="absolute",o.style.top=`${h}px`,o.style.left=`${m}px`,o.style.zIndex="10001",console.log("📝 Tooltip positioned:",{highlightRect:l,tooltipRect:d,finalPosition:{top:h,left:m}}),o.style.opacity="0",o.style.transition="opacity 0.2s ease",setTimeout(()=>{o&&(o.style.opacity="1")},10)}},10)},100))},a=()=>{s&&(clearTimeout(s),s=null),n=setTimeout(()=>{o&&(o.style.opacity="0",setTimeout(()=>{o&&(o.remove(),o=null)},200))},500)};e.addEventListener("mouseenter",i),e.addEventListener("mouseleave",a)}catch(o){console.error("❌ Failed to add hover tooltip:",o)}}async resolveComment(e,t){try{if(console.log("📝 Resolving comment:",e,t),this.resolvingComments&&this.resolvingComments.has(e)){console.log("📝 Comment resolution already in progress:",e);return}this.resolvingComments||(this.resolvingComments=new Set),this.resolvingComments.add(e);try{this.commentManager?await this.commentManager.resolveComment(e):console.warn("⚠️ Comment manager not available for resolve")}finally{this.resolvingComments.delete(e)}}catch(o){console.error("❌ Failed to resolve comment:",o),this.resolvingComments&&this.resolvingComments.delete(e)}}setCommentManager(e){this.commentManager=e}async resolveHighlight(e){try{this.highlights.get(e)&&(await this.removeHighlight(e),console.log("📝 Highlight resolved and removed:",e))}catch(t){console.error("❌ Failed to resolve highlight:",t)}}async removeHighlight(e){try{const t=this.highlights.get(e);if(t){const o=t.parentNode;for(;t.firstChild;)o.insertBefore(t.firstChild,t);o.removeChild(t),this.highlights.delete(e);for(const[s,n]of this.commentHighlights)if(n===e){this.commentHighlights.delete(s);break}console.log("📝 Highlight removed:",e)}}catch(t){console.error("❌ Failed to remove highlight:",t)}}getHighlight(e){return this.highlights.get(e)}getHighlightByCommentId(e){const t=this.commentHighlights.get(e);return t?this.highlights.get(t):null}getAllHighlights(){return Array.from(this.highlights.values())}async clearAllHighlights(){try{for(const[e,t]of this.highlights)await this.removeHighlight(e);this.highlights.clear(),this.commentHighlights.clear(),console.log("📝 All highlights cleared")}catch(e){console.error("❌ Failed to clear all highlights:",e)}}async clearHighlightsForRoomChange(){try{console.log("📝 Clearing highlights for room change...");for(const[e,t]of this.highlights)if(t&&t.parentNode){const o=t.parentNode;for(;t.firstChild;)o.insertBefore(t.firstChild,t);o.removeChild(t)}this.highlights.clear(),this.commentHighlights.clear(),console.log("📝 Highlights cleared for room change")}catch(e){console.error("❌ Failed to clear highlights for room change:",e)}}createClickableAuthor(e){return e.author==="You"?e.author:e.authorUrl?`<a href="${e.authorUrl}" target="_blank" class="comment-author-link" title="${e.authorTitle||"View user page"}">${e.author}</a>`:e.author}formatTimestamp(e){const t=new Date(e),s=new Date-t,n=Math.floor(s/6e4),i=Math.floor(s/36e5),a=Math.floor(s/864e5);return n<1?"Just now":n<60?`${n}m ago`:i<24?`${i}h ago`:a<7?`${a}d ago`:t.toLocaleDateString()}escapeHtml(e){const t=document.createElement("div");return t.textContent=e,t.innerHTML}destroy(){try{this.clearAllHighlights(),console.log("📝 CommentHighlights destroyed")}catch(e){console.error("❌ Failed to destroy CommentHighlights:",e)}}}class O{constructor(e={}){this.adapter=e.adapter||null,this.name=e.name||"default-db",this.version=e.version||1,this.isInitialized=!1,console.log("🗄️ Database initialized:",this.name)}setAdapter(e){this.adapter=e,console.log("🗄️ Database adapter set:",e.constructor.name)}async initialize(){if(!this.adapter)throw new Error("No adapter set for database");await this.adapter.initialize({name:this.name,version:this.version}),this.isInitialized=!0,console.log("🗄️ Database initialized successfully")}async create(e,t){return this.ensureInitialized(),await this.adapter.create(e,t)}async read(e,t){return this.ensureInitialized(),await this.adapter.read(e,t)}async update(e,t,o){return this.ensureInitialized(),await this.adapter.update(e,t,o)}async delete(e,t){return this.ensureInitialized(),await this.adapter.delete(e,t)}async getAll(e){return this.ensureInitialized(),await this.adapter.getAll(e)}async query(e,t,o){return this.ensureInitialized(),await this.adapter.query(e,t,o)}async clear(e){return this.ensureInitialized(),await this.adapter.clear(e)}async count(e){return this.ensureInitialized(),await this.adapter.count(e)}ensureInitialized(){if(!this.isInitialized)throw new Error("Database not initialized. Call initialize() first.")}async close(){this.adapter&&this.adapter.close&&await this.adapter.close(),this.isInitialized=!1,console.log("🗄️ Database closed")}}class W{constructor(){this.db=null,this.stores=new Map}async initialize(e){const{name:t,version:o}=e;return new Promise((s,n)=>{const i=indexedDB.open(t,o);i.onerror=()=>{n(new Error(`Failed to open IndexedDB database: ${t}`))},i.onsuccess=()=>{this.db=i.result,console.log("🗄️ IndexedDB database opened:",t),s()},i.onupgradeneeded=a=>{const r=a.target.result;this.createStores(r)}})}defineStore(e,t={}){this.stores.set(e,{keyPath:t.keyPath||"id",indexes:t.indexes||[],autoIncrement:t.autoIncrement||!1})}createStores(e){for(const[t,o]of this.stores)if(!e.objectStoreNames.contains(t)){const s=e.createObjectStore(t,{keyPath:o.keyPath,autoIncrement:o.autoIncrement});for(const n of o.indexes)s.createIndex(n.name,n.keyPath,{unique:n.unique||!1});console.log("🗄️ Created store:",t)}}async create(e,t){return new Promise((o,s)=>{const a=this.db.transaction([e],"readwrite").objectStore(e).add(t);a.onsuccess=()=>{console.log("🗄️ Record created in store:",e),o(t)},a.onerror=()=>{s(new Error(`Failed to create record in store: ${e}`))}})}async read(e,t){return new Promise((o,s)=>{const a=this.db.transaction([e],"readonly").objectStore(e).get(t);a.onsuccess=()=>{o(a.result)},a.onerror=()=>{s(new Error(`Failed to read record from store: ${e}`))}})}async update(e,t,o){return new Promise((s,n)=>{const r=this.db.transaction([e],"readwrite").objectStore(e).put({...o,id:t});r.onsuccess=()=>{console.log("🗄️ Record updated in store:",e),s({...o,id:t})},r.onerror=()=>{n(new Error(`Failed to update record in store: ${e}`))}})}async delete(e,t){return new Promise((o,s)=>{const a=this.db.transaction([e],"readwrite").objectStore(e).delete(t);a.onsuccess=()=>{console.log("🗄️ Record deleted from store:",e),o()},a.onerror=()=>{s(new Error(`Failed to delete record from store: ${e}`))}})}async getAll(e){return new Promise((t,o)=>{const i=this.db.transaction([e],"readonly").objectStore(e).getAll();i.onsuccess=()=>{t(i.result)},i.onerror=()=>{o(new Error(`Failed to get all records from store: ${e}`))}})}async query(e,t,o){return new Promise((s,n)=>{const c=this.db.transaction([e],"readonly").objectStore(e).index(t).getAll(o);c.onsuccess=()=>{s(c.result)},c.onerror=()=>{n(new Error(`Failed to query records from store: ${e}, index: ${t}`))}})}async clear(e){return new Promise((t,o)=>{const i=this.db.transaction([e],"readwrite").objectStore(e).clear();i.onsuccess=()=>{console.log("🗄️ Store cleared:",e),t()},i.onerror=()=>{o(new Error(`Failed to clear store: ${e}`))}})}async count(e){return new Promise((t,o)=>{const i=this.db.transaction([e],"readonly").objectStore(e).count();i.onsuccess=()=>{t(i.result)},i.onerror=()=>{o(new Error(`Failed to count records in store: ${e}`))}})}async close(){this.db&&(this.db.close(),this.db=null,console.log("🗄️ IndexedDB connection closed"))}}class G{constructor(e={}){this.db=new O({name:e.dbName||"commentDB",version:e.version||1}),this.adapter=new W,this.db.setAdapter(this.adapter),this.adapter.defineStore("comments",{keyPath:"id",indexes:[{name:"url",keyPath:"url",unique:!1},{name:"room",keyPath:"room",unique:!1},{name:"highlightId",keyPath:"highlightId",unique:!1},{name:"peerId",keyPath:"peerId",unique:!1},{name:"timestamp",keyPath:"timestamp",unique:!1},{name:"resolved",keyPath:"resolved",unique:!1}]}),this.isInitialized=!1,console.log("📝 CommentDB initialized")}async initialize(){this.isInitialized||(await this.db.initialize(),this.isInitialized=!0,console.log("📝 CommentDB fully initialized"))}async saveComment(e){return await this.initialize(),await this.db.create("comments",e)}async getCommentById(e){return await this.initialize(),await this.db.read("comments",e)}async getCommentsByUrlAndRoom(e,t){return await this.initialize(),(await this.db.query("comments","url",e)).filter(s=>s.room===t)}async getCommentsByHighlightId(e){return await this.initialize(),await this.db.query("comments","highlightId",e)}async getCommentsByPeerId(e){return await this.initialize(),await this.db.query("comments","peerId",e)}async getUnresolvedComments(e,t){return(await this.getCommentsByUrlAndRoom(e,t)).filter(s=>!s.resolved)}async deleteComment(e){return await this.initialize(),await this.db.delete("comments",e)}async resolveComment(e,t){await this.initialize();const o=await this.getCommentById(e);if(!o)throw new Error("Comment not found");return o.resolved=!0,o.resolvedBy=t,o.resolvedAt=Date.now(),await this.db.update("comments",e,o)}async getCommentStats(e,t){const o=await this.getCommentsByUrlAndRoom(e,t),s=o.filter(i=>i.resolved).length,n=o.length-s;return{total:o.length,resolved:s,unresolved:n,byPeer:this.groupCommentsByPeer(o)}}groupCommentsByPeer(e){const t={};for(const o of e){const s=o.peerId;t[s]||(t[s]={total:0,resolved:0,unresolved:0}),t[s].total++,o.resolved?t[s].resolved++:t[s].unresolved++}return t}async clearCommentsForRoom(e){await this.initialize();const t=await this.db.query("comments","room",e);for(const o of t)await this.db.delete("comments",o.id);console.log(`📝 Cleared ${t.length} comments for room: ${e}`)}async exportComments(e,t){const o=await this.getCommentsByUrlAndRoom(e,t);return{url:e,room:t,exportedAt:Date.now(),comments:o}}async importComments(e){await this.initialize();for(const t of e.comments)t.room=e.room,await this.db.create("comments",t);console.log(`📝 Imported ${e.comments.length} comments`)}async close(){await this.db.close(),this.isInitialized=!1,console.log("📝 CommentDB closed")}}class K{constructor(e={}){this.currentRoom=e.currentRoom||"default",this.webrtcChat=e.webrtcChat||null,this.userCursors=e.userCursors||new Map,this.ui=new B({onCommentSave:this.handleCommentSave.bind(this),onCommentDelete:this.handleCommentDelete.bind(this),onCommentResolve:this.handleCommentResolve.bind(this)}),this.data=new j({currentRoom:this.currentRoom}),this.commentDB=new G({dbName:"commentDB",version:1}),this.sync=new Q({webrtcChat:this.webrtcChat,onCommentAdd:this.handleCommentAdd.bind(this),onCommentDelete:this.handleCommentDelete.bind(this),onCommentResolve:this.handleCommentResolve.bind(this),onCommentStatusRequest:this.handleCommentStatusRequest.bind(this),onCommentStatusResponse:this.handleCommentStatusResponse.bind(this)}),this.highlights=new _({currentRoom:this.currentRoom}),this.hasComments=new Set,this.processedComments=new Set,this.isInitialized=!1,this.processedCommentsCleanupInterval=setInterval(()=>{this.processedComments.size>1e3&&(console.log("📝 Cleaning up processed comments cache"),this.processedComments.clear())},6e4),console.log("📝 CommentManager initialized")}async initialize(){try{await this.data.initialize(),await this.commentDB.initialize(),await this.loadExistingComments(),this.ui.initialize(),this.highlights.initialize(),this.highlights.setCommentManager(this),this.isInitialized=!0,console.log("📝 CommentManager fully initialized")}catch(e){throw console.error("❌ Failed to initialize CommentManager:",e),e}}updateWebRTCChat(e){this.webrtcChat=e,this.sync.updateWebRTCChat(e)}async updateCurrentRoom(e){try{console.log("📝 Updating room from",this.currentRoom,"to",e),await this.highlights.clearHighlightsForRoomChange(),this.hasComments.clear(),this.processedComments.clear(),this.currentRoom=e,this.data.updateCurrentRoom(e),this.highlights.updateCurrentRoom(e),await this.loadCommentsForRoom(),console.log("📝 Room updated successfully to:",e)}catch(t){console.error("❌ Failed to update room:",t)}}updateUserCursors(e){this.userCursors=e}async createCommentFromSelection(e){try{if(!this.isInitialized){console.warn("⚠️ CommentManager not initialized");return}await this.ui.showCommentPanel(e)}catch(t){console.error("❌ Failed to create comment from selection:",t)}}async handleCommentSave(e,t){try{console.log("📝 CommentManager.handleCommentSave called with:",{commentText:e,selectionData:t});const o=this.webrtcChat?.myClientId||"unknown",s={id:`${o}-${Date.now()}-${Math.random().toString(36).substr(2,9)}`,highlightId:`highlight-${o}-${Date.now()}-${Math.random().toString(36).substr(2,9)}`,peerId:o,text:e,url:t.url||window.location.href,room:this.currentRoom,highlightData:t,timestamp:Date.now(),author:this.getAuthorName(o),authorUrl:this.getAuthorUrl(o),authorTitle:this.getAuthorTitle(o),resolved:!1};console.log("📝 Created comment object:",s),this.processedComments.add(s.id),await this.data.saveComment(s);const n=await this.highlights.createHighlightFromComment(s);this.highlights.removePlaceholderHighlights(),await this.sync.broadcastCommentAdd(s),await this.sync.broadcastSystemMessage(s),this.hasComments.add(s.highlightId),o===this.webrtcChat?.myClientId&&this.incrementUserAction("comment"),console.log("📝 Comment saved:",s.id)}catch(o){console.error("❌ Failed to save comment:",o)}}async handleCommentAdd(e){try{if(!this.validateRoomAndPage(e.url,e.room))return;if(this.processedComments.has(e.id)){console.log("📝 Comment already processed:",e.id);return}this.processedComments.add(e.id),await this.data.saveComment(e);const t=e.peerId,o=this.userCursors.get(t)||{};this.userCursors.set(t,{...o,lastAction:"Added a comment"}),e.resolved?console.log("📝 Skipping highlight creation for resolved comment:",e.id):(await this.highlights.createHighlightFromComment(e),this.hasComments.add(e.highlightId)),console.log("📝 Received comment from peer:",e)}catch(t){console.error("❌ Failed to process incoming comment:",t)}}async handleCommentDelete(e){try{await this.data.deleteComment(e);const t=await this.data.getCommentById(e);t&&((await this.data.getCommentsByUrlAndRoom(window.location.href,this.currentRoom)).some(n=>n.highlightId===t.highlightId&&n.id!==e)||console.log(`📝 No more comments for highlight ${t.highlightId} - highlight remains for reference`)),console.log("📝 Comment deleted:",e)}catch(t){console.error("❌ Failed to delete comment:",t)}}async handleCommentResolve(e,t,o,s,n){try{if(!this.validateRoomAndPage(o,s))return;const i=await this.data.getCommentById(e);if(i&&i.resolved){console.log("📝 Comment already resolved (sync):",e);return}await this.data.resolveComment(e,n),await this.highlights.resolveHighlight(t),console.log("📝 Comment resolved:",e)}catch(i){console.error("❌ Failed to resolve comment:",i)}}async handleCommentStatusRequest(e,t){try{const s=(await this.data.getCommentsByUrlAndRoom(e.url,e.room)).map(i=>i.id),n=e.commentIds.filter(i=>!s.includes(i));n.length>0&&await this.sync.broadcastCommentStatusResponse(n,e.url,e.room)}catch(o){console.error("❌ Failed to handle comment status request:",o)}}async handleCommentStatusResponse(e,t){try{console.log("📝 Received comment status response:",e)}catch(o){console.error("❌ Failed to handle comment status response:",o)}}async loadExistingComments(){try{const e=await this.data.getCommentsByUrlAndRoom(window.location.href,this.currentRoom);for(const t of e)t.resolved?console.log("📝 Skipping resolved comment:",t.id):(await this.highlights.createHighlightFromComment(t),this.hasComments.add(t.highlightId));console.log(`📝 Loaded ${e.length} existing comments`)}catch(e){console.error("❌ Failed to load existing comments:",e)}}async loadCommentsForRoom(){try{if(!this.currentRoom){console.warn("⚠️ No current room set, skipping comment loading");return}console.log("📝 Loading comments for room:",this.currentRoom);const e=await this.data.getCommentsByUrlAndRoom(window.location.href,this.currentRoom);for(const t of e)t.resolved?console.log("📝 Skipping resolved comment:",t.id):(await this.highlights.createHighlightFromComment(t),this.hasComments.add(t.highlightId));console.log(`📝 Loaded ${e.length} comments for room "${this.currentRoom}"`),this.logRoomIsolationStatus()}catch(e){console.error("❌ Failed to load comments for room:",e)}}logRoomIsolationStatus(){const e=this.highlights.getAllHighlights().length,t=this.hasComments.size;console.log("📝 Room isolation status:",{currentRoom:this.currentRoom,currentPage:window.location.href,highlightsInDOM:e,trackedComments:t,isolationWorking:e===t})}requestCommentStatus(e){this.webrtcChat&&e.length>0&&this.sync.broadcastCommentStatusRequest(e)}async resolveComment(e){try{const t=await this.data.getCommentById(e);if(!t){console.warn("⚠️ Comment not found:",e);return}if(t.resolved){console.log("📝 Comment already resolved:",e);return}await this.data.resolveComment(e,this.webrtcChat?.myClientId),await this.highlights.resolveHighlight(t.highlightId),await this.sync.broadcastCommentResolve(e,t.highlightId,t.url,t.room,this.webrtcChat?.myClientId),console.log(`📝 Comment ${e} resolved.`)}catch(t){console.error("❌ Failed to resolve comment:",t)}}getAuthorName(e){return this.webrtcChat?.myClientId===e?"You":this.webrtcChat?.peers&&this.webrtcChat.peers[e]?this.webrtcChat.peers[e].username||`User ${e.substring(0,8)}`:`User ${e.substring(0,8)}`}getAuthorUrl(e){return this.webrtcChat?.myClientId===e?window.location.href:this.webrtcChat?.peers&&this.webrtcChat.peers[e]&&this.webrtcChat.peers[e].url||null}getAuthorTitle(e){return this.webrtcChat?.myClientId===e?document.title:this.webrtcChat?.peers&&this.webrtcChat.peers[e]&&this.webrtcChat.peers[e].title||null}validateRoomAndPage(e,t){const o=e===window.location.href&&t===this.currentRoom;return o||console.log("📝 Comment operation not for current page/room:",{provided:{url:e,room:t},current:{url:window.location.href,room:this.currentRoom}}),o}incrementUserAction(e){console.log(`📊 User action: ${e}`)}async getCommentStats(){try{const e=await this.data.getCommentsByUrlAndRoom(window.location.href,this.currentRoom),t=e.filter(s=>s.resolved).length,o=e.length-t;return{total:e.length,resolved:t,unresolved:o,highlights:this.hasComments.size}}catch(e){return console.error("❌ Failed to get comment stats:",e),{total:0,resolved:0,unresolved:0,highlights:0}}}destroy(){try{this.ui.destroy(),this.data.destroy(),this.sync.destroy(),this.highlights.destroy(),this.hasComments.clear(),this.processedComments.clear(),this.processedCommentsCleanupInterval&&(clearInterval(this.processedCommentsCleanupInterval),this.processedCommentsCleanupInterval=null),this.isInitialized=!1,console.log("📝 CommentManager destroyed")}catch(e){console.error("❌ Failed to destroy CommentManager:",e)}}}class A{constructor(e={}){const o=window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1"||window.location.hostname==="0.0.0.0"?"ws://localhost:8789":"wss://sync-server.mlsysbook.workers.dev";this.options={signalingServerUrl:e.signalingServerUrl||o,username:e.username||null,enableChat:e.enableChat!==!1,enablePresence:e.enablePresence!==!1,...e},this.webrtcChat=null,this.plugins=new Map,this.gamificationComponent=null,this.widgetContainer=null,this.isMinimized=!1,this.currentToolbar=null,this.isConnected=!1,this.lastStatsKey=null,this.currentRoom=null,this.hasComments=new Set,this.commentDB=null,this.commentManager=null,this.quizManager=null,this.storageKey="collaborative-widget-session",this.autoReconnectEnabled=!0,this.persistentUserId=this.getPersistentUserId(),this.userStats=this.loadUserStats(),this.initializeWidget()}exposeGlobalAPI(){window.CollaborativeWidgetAPI||(window.CollaborativeWidgetAPI={}),window.CollaborativeWidgetAPI.instance=this,window.CollaborativeWidgetAPI.show=()=>this.showWidget(),window.CollaborativeWidgetAPI.hide=()=>this.hideWidget(),window.CollaborativeWidgetAPI.hideTemporarily=()=>this.hideWidgetTemporarily(),window.CollaborativeWidgetAPI.toggle=()=>this.toggleWidget(),window.CollaborativeWidgetAPI.isVisible=()=>this.isWidgetVisible(),window.CollaborativeWidgetAPI.connect=(e,t,o)=>this.connectToRoom(e,t,o),window.CollaborativeWidgetAPI.disconnect=()=>this.disconnectFromRoom(),console.log("🔧 CollaborativeWidget Global API exposed to window.CollaborativeWidgetAPI"),console.log("Available methods: show(), hide(), hideTemporarily(), toggle(), isVisible(), connect(username, roomName, password), disconnect()")}showWidget(){this.shadowHost&&(this.shadowHost.style.display="block",console.log("✅ Widget shown"))}hideWidget(){this.shadowHost&&(this.shadowHost.style.display="none",console.log("❌ Widget hidden")),this.disconnectFromRoom()}hideWidgetTemporarily(){this.shadowHost&&(this.shadowHost.style.display="none",console.log("👁️ Widget temporarily hidden (connections remain active)"))}toggleWidget(){this.isWidgetVisible()?this.hideWidget():this.showWidget()}isWidgetVisible(){return this.shadowHost&&this.shadowHost.style.display!=="none"}async connectToRoom(e,t,o=null){if(!e||!t)return console.error("❌ Username and room name are required"),!1;try{return await this.joinRoom(e,t,o),this.showWidget(),!0}catch(s){return console.error("❌ Failed to connect to room:",s),!1}}disconnectFromRoom(){this.webrtcChat&&this.webrtcChat.disconnect(),this.clearRoomSession(),console.log("🔌 Disconnected from room")}generateRandomUsername(){const e=["Curious","Smart","Focused","Creative","Analytical","Innovative","Brilliant","Sharp","Wise","Insightful"],t=["Researcher","Scholar","Explorer","Thinker","Learner","Analyst","Scientist","Expert","Guru","Master"];return`${e[Math.floor(Math.random()*e.length)]}${t[Math.floor(Math.random()*t.length)]}`}getPersistentUserId(){let e=localStorage.getItem("collaborative-widget-persistent-user-id");return e?console.log("Retrieved existing persistent user ID:",e):(e="user_"+Date.now()+"_"+Math.random().toString(36).substr(2,9),localStorage.setItem("collaborative-widget-persistent-user-id",e),console.log("Generated new persistent user ID:",e)),e}loadUserStats(){const e=`collaborative-widget-user-stats-${this.persistentUserId}`,t=localStorage.getItem(e);if(t)return JSON.parse(t);const o={persistentId:this.persistentUserId,totalStars:0,totalScore:0,badges:[],achievements:[],roomScores:{},joinDate:Date.now(),lastSeen:Date.now(),totalMessages:0,totalComments:0,totalActions:0};return this.saveUserStats(o),o}saveUserStats(e){const t=`collaborative-widget-user-stats-${this.persistentUserId}`;e.lastSeen=Date.now(),localStorage.setItem(t,JSON.stringify(e))}updateUserScore(e,t=null){this.userStats.totalScore+=e,this.userStats.totalStars+=e,this.userStats.totalStars!==this.userStats.totalScore&&(console.warn(`Stars/Score mismatch detected! Aligning: stars=${this.userStats.totalStars}, totalScore=${this.userStats.totalScore}`),this.userStats.totalStars=this.userStats.totalScore),t&&(this.userStats.roomScores[t]||(this.userStats.roomScores[t]=0),this.userStats.roomScores[t]+=e),this.saveUserStats(this.userStats),console.log(`Updated user score: +${e} (Total: ${this.userStats.totalScore})`);const o=this.shadowRoot.getElementById("user-stats-content");o&&(this.statEffects.animateStatUpdate(o,"total-score",this.userStats.totalScore),t&&this.statEffects.animateStatUpdate(o,"room-score",this.userStats.roomScores[t]))}addUserBadge(e){this.userStats.badges.find(t=>t.icon===e.icon&&t.text===e.text)||(this.userStats.badges.push(e),this.saveUserStats(this.userStats),console.log("Added badge:",e))}incrementUserAction(e){if(this.userStats.totalActions++,e==="message"){this.userStats.totalMessages++;const t=this.shadowRoot.getElementById("user-stats-content");t&&this.statEffects.animateStatUpdate(t,"messages",this.userStats.totalMessages)}else if(e==="comment"){this.userStats.totalComments++;const t=this.shadowRoot.getElementById("user-stats-content");t&&this.statEffects.animateStatUpdate(t,"comments",this.userStats.totalComments)}this.saveUserStats(this.userStats)}awardQuizPoint(e=null){this.updateUserScore(1,e),this.checkBadgeMilestones();const t=this.webrtcChat.myClientId;if(t){const o=this.userCursors.get(t)||{};this.userCursors.set(t,{...o,lastAction:"Answered quiz correctly",stars:this.userStats.totalStars,totalScore:this.userStats.totalScore,badges:this.userStats.badges})}console.log(`Quiz point awarded! Total score: ${this.userStats.totalScore}`),this.gamificationComponent&&this.updatePresenceList()}checkBadgeMilestones(){const e=this.userStats.totalScore,t=this.userStats.badges;[{points:1,emoji:"🌱",name:"First Bloom",description:"Your first correct answer!"},{points:5,emoji:"⭐",name:"Rising Star",description:"5 correct answers!"},{points:10,emoji:"🌟",name:"Bright Spark",description:"10 correct answers!"},{points:15,emoji:"💫",name:"Shooting Star",description:"15 correct answers!"},{points:20,emoji:"✨",name:"Sparkling Mind",description:"20 correct answers!"},{points:25,emoji:"🎯",name:"Precision Master",description:"25 correct answers!"},{points:30,emoji:"🏆",name:"Quiz Champion",description:"30 correct answers!"},{points:35,emoji:"👑",name:"Knowledge Royalty",description:"35 correct answers!"},{points:40,emoji:"🧠",name:"Brain Genius",description:"40 correct answers!"},{points:45,emoji:"🎓",name:"Scholar Supreme",description:"45 correct answers!"},{points:50,emoji:"💎",name:"Diamond Mind",description:"50 correct answers!"}].forEach(s=>{if(e>=s.points&&!t.find(i=>i.points===s.points)){const i={emoji:s.emoji,name:s.name,description:s.description,points:s.points,earnedAt:Date.now()};t.push(i),this.saveUserStats(this.userStats),this.showBadgeNotification(i),console.log(`Badge earned: ${s.emoji} ${s.name} (${s.points} points)`)}})}showBadgeNotification(e){const t=document.createElement("div");t.className="badge-notification",t.innerHTML=`
            <div class="badge-notification-content">
                <div class="badge-emoji">${e.emoji}</div>
                <div class="badge-info">
                    <div class="badge-name">${e.name}</div>
                    <div class="badge-description">${e.description}</div>
                </div>
            </div>
        `,t.style.cssText=`
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            z-index: 10001;
            animation: badgeSlideIn 0.5s ease-out;
            max-width: 300px;
        `;const o=document.createElement("style");o.textContent=`
            @keyframes badgeSlideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            .badge-notification-content {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .badge-emoji {
                font-size: 24px;
                animation: badgeBounce 0.6s ease-in-out;
            }
            .badge-name {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 2px;
            }
            .badge-description {
                font-size: 12px;
                opacity: 0.9;
            }
            @keyframes badgeBounce {
                0%, 20%, 50%, 80%, 100% {
                    transform: translateY(0);
                }
                40% {
                    transform: translateY(-10px);
                }
                60% {
                    transform: translateY(-5px);
                }
            }
        `,document.head.appendChild(o),document.body.appendChild(t),setTimeout(()=>{t.style.animation="badgeSlideIn 0.3s ease-in reverse",setTimeout(()=>{t.parentNode&&t.parentNode.removeChild(t),o.parentNode&&o.parentNode.removeChild(o)},300)},4e3)}updateUserStatsDisplay(){const e=this.shadowRoot.getElementById("user-stats-content"),t=this.shadowRoot.getElementById("user-stats-title");if(!e){console.warn("Stats content element not found");return}if(console.log("Updating user stats display:",{userStats:this.userStats,currentRoom:this.currentRoom,isConnected:this.isConnected,username:this.options.username}),t){const o=this.statEffects.getRandomGreeting(this.options.username);t.textContent=o}this.statEffects.updateStatsDisplay(e,this.userStats,this.currentRoom,this.isConnected,this.options.username)}updateUserStatsDisplayWithAnimation(){const e=this.shadowRoot.getElementById("user-stats-content"),t=this.shadowRoot.getElementById("user-stats-title");if(!e){console.warn("Stats content element not found");return}if(console.log("Updating user stats display with animation:",{userStats:this.userStats,currentRoom:this.currentRoom,isConnected:this.isConnected,username:this.options.username}),t){const o=this.statEffects.getRandomGreeting(this.options.username);t.textContent=o}this.statEffects.animateStatsDisplay(e,this.userStats,this.currentRoom,this.isConnected,this.options.username)}updateUserStatsDisplayIfChanged(){const e=JSON.stringify({totalScore:this.userStats.totalScore,totalMessages:this.userStats.totalMessages,totalComments:this.userStats.totalComments,currentRoom:this.currentRoom,isConnected:this.isConnected});(this.lastStatsKey===null||this.lastStatsKey!==e)&&(this.lastStatsKey=e,this.updateUserStatsDisplay())}saveRoomSession(e,t,o,s){const n={username:e,roomName:t,password:o,mode:s,timestamp:Date.now(),domain:window.location.hostname};localStorage.setItem(this.storageKey,JSON.stringify(n)),this.saveFormValues(e,t)}saveFormValues(e,t){const o={username:e,roomName:t,timestamp:Date.now()};localStorage.setItem("collaborative-widget-form-data",JSON.stringify(o))}loadFormValues(){try{const e=localStorage.getItem("collaborative-widget-form-data");if(!e)return null;const t=JSON.parse(e);return Date.now()-t.timestamp<30*24*60*60*1e3?t:(localStorage.removeItem("collaborative-widget-form-data"),null)}catch(e){return console.error("Failed to load form values:",e),localStorage.removeItem("collaborative-widget-form-data"),null}}loadRoomSession(){try{const e=localStorage.getItem(this.storageKey);if(!e)return null;const t=JSON.parse(e),o=t.domain===window.location.hostname,s=Date.now()-t.timestamp<24*60*60*1e3;return o&&s?t:(localStorage.removeItem(this.storageKey),null)}catch(e){return console.error("Failed to load room session:",e),localStorage.removeItem(this.storageKey),null}}clearRoomSession(){localStorage.removeItem(this.storageKey)}saveChatState(){const e={isOpen:this.shadowRoot.getElementById("widget-panel").style.display!=="none",isMinimized:this.isMinimized,timestamp:Date.now()};localStorage.setItem("collaborative-widget-chat-state",JSON.stringify(e))}loadChatState(){try{const e=localStorage.getItem("collaborative-widget-chat-state");if(!e)return null;const t=JSON.parse(e);return Date.now()-t.timestamp<7*24*60*60*1e3?t:(localStorage.removeItem("collaborative-widget-chat-state"),null)}catch(e){return console.error("Failed to load chat state:",e),localStorage.removeItem("collaborative-widget-chat-state"),null}}restoreChatState(){const e=this.loadChatState();if(!e)return;const t=this.shadowRoot.getElementById("widget-panel"),o=this.shadowRoot.getElementById("widget-toggle");e.isOpen&&!e.isMinimized?(t.style.display="block",o.style.display="none",this.isMinimized=!1):e.isMinimized&&(t.style.display="none",o.style.display="flex",this.isMinimized=!0)}async attemptAutoReconnect(){const e=this.loadRoomSession();if(!e||!this.autoReconnectEnabled)return!1;try{console.log("Attempting auto-reconnect to room:",e.roomName),this.options.username=e.username,this.currentRoom=e.roomName,this.webrtcChat=new w({signalingServerUrl:this.options.signalingServerUrl,username:this.options.username,onConnectionStatusChange:this.onConnectionStatusChange.bind(this),onUserCountChange:this.onUserCountChange.bind(this),onMessage:this.onMessage.bind(this),onInfoMessage:this.onInfoMessage.bind(this),onTypingIndicatorChange:this.onTypingIndicatorChange.bind(this),onUserJoined:this.onUserJoined.bind(this),onUserLeft:this.onUserLeft.bind(this),onCommentAdd:this.onCommentAdd.bind(this),onCommentDelete:this.onCommentDelete.bind(this),onCommentResolve:this.onCommentResolve.bind(this),onCommentStatusRequest:this.onCommentStatusRequest.bind(this),onCommentStatusResponse:this.onCommentStatusResponse.bind(this),onSystemMessage:this.onSystemMessage.bind(this),onCursorUpdate:this.handleCursorUpdate.bind(this),onPresenceUpdate:this.handlePresenceUpdate.bind(this)}),console.log("🎮 Auto-reconnect: Setting webrtcChat reference in gamification component:",this.webrtcChat),this.gamificationComponent.updateWebRTCChat(this.webrtcChat),console.log("🎮 Auto-reconnect: Gamification webrtcChat after assignment:",this.gamificationComponent.webrtcChat),await this.webrtcChat.joinRoom(e.username,e.roomName,e.password,e.mode,!0),await new Promise(o=>setTimeout(o,1e3)),this.showChatInterface(),this.quizManager=new C(this),this.loadPlugins(),await this.initializeCommentSystem(),this.setupGlobalEventListeners(),this.loadExistingComments(),this.showAIUsageHelp(),this.userCursors.set(this.webrtcChat.myClientId,{persistentId:this.persistentUserId,sessionId:this.webrtcChat.myClientId,username:this.options.username,badges:this.userStats.badges.length>0?this.userStats.badges:this._getRandomBadges(),stars:this.userStats.totalStars,totalScore:this.userStats.totalScore,lastAction:"Reconnected to room",joinDate:this.userStats.joinDate,totalMessages:this.userStats.totalMessages,totalComments:this.userStats.totalComments});const t=this.shadowRoot.getElementById("gamification-btn");return t&&(t.disabled=!1,t.classList.add("glowing-border")),this.restoreChatState(),console.log("Auto-reconnect successful"),!0}catch(t){return console.error("Auto-reconnect failed:",t),this.clearRoomSession(),!1}}async initializeWidget(){this.createWidgetContainer(),this.setupThemeObserver(),this.injectGlobalStyles(),this.injectMarkdownStyles(),this.injectComponentStyles(),await this.attemptAutoReconnect()||this.showRoomSelection(),this.options.startHidden&&this.hideWidget(),this.exposeGlobalAPI()}createWidgetContainer(){this.shadowHost=document.createElement("div"),this.shadowHost.id="collaborative-widget-host",this.shadowHost.style.position="fixed",this.shadowHost.style.zIndex="10000",document.body.appendChild(this.shadowHost),this.shadowRoot=this.shadowHost.attachShadow({mode:"open"}),this.shadowRoot.innerHTML=z;const e=document.createElement("style");e.textContent=`
          ${T}
        `,this.shadowRoot.prepend(e);const t=document.createElement("style");t.textContent=M,this.shadowRoot.prepend(t),setTimeout(()=>{this.shadowHost.style.opacity="1",console.log("🎯 Widget shadow DOM created:",this.shadowHost),console.log("🎯 Shadow root:",this.shadowRoot),console.log("🎯 Widget toggle element:",this.shadowRoot.getElementById("widget-toggle"))},1e3),this.setupWidgetControls();const o=this.shadowRoot.getElementById("gamification-btn");this.gamificationComponent=new E(this.shadowRoot,this.webrtcChat,o,this),this.statEffects=new $,this.initResizer()}initResizer(){const e=this.shadowRoot.getElementById("widget-panel"),t=this.shadowRoot.querySelector(".resize-handle-ne");e&&t&&new R(e,t)}showRoomSelection(){const e=this.loadRoomSession(),t=this.loadFormValues(),o=e!==null,s=this.options.username||(o?e.username:t?t.username:""),n=o?e.roomName:t?t.roomName:"",i=this.shadowRoot.getElementById("widget-content");i.innerHTML=`
            <div class="room-selection">
                <div class="room-header">
                    <h3>Join Collaboration</h3>
                    <p>Choose a room to collaborate in</p>
                    ${o?`
                        <div class="auto-reconnect-info" style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 12px; margin: 12px 0; font-size: 14px;">
                            <div style="display: flex; align-items: center; gap: 8px; color: #1976d2;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path>
                                </svg>
                                <strong>Auto-reconnect available</strong>
                            </div>
                            <div style="margin-top: 4px; color: #424242;">
                                Last session: ${e.username} in "${e.roomName}"
                            </div>
                            <button type="button" id="auto-reconnect-btn" style="margin-top: 8px; background: #2196f3; color: white; border: none; border-radius: 4px; padding: 6px 12px; font-size: 12px; cursor: pointer;">
                                Reconnect Automatically
                            </button>
                        </div>
                    `:""}
                </div>
                
                <div class="room-tabs">
                    <button class="room-tab active" data-mode="join">Join Room</button>
                    <button class="room-tab" data-mode="create">Create Room</button>
                </div>
                
                <form class="room-form">
                    <div class="form-group">
                        <label for="username">Your Name</label>
                        <input type="text" id="username" placeholder="Enter your name" value="${s}" autocomplete="username" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="room-name">Room Name</label>
                        <input type="text" id="room-name" placeholder="Enter room name" value="${n}" autocomplete="off" required>
                    </div>
                    
                    <div class="form-group password-group" style="display: none;">
                        <label for="room-password">Password (Optional)</label>
                        <input type="password" id="room-password" placeholder="Enter password" autocomplete="new-password">
                    </div>
                    
                    <div class="form-group join-password-group" style="display: none;">
                        <label for="join-password">Room Password</label>
                        <input type="password" id="join-password" placeholder="Enter room password" autocomplete="current-password">
                    </div>
                    
                    <button type="button" class="connect-btn" id="connect-btn">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="8.5" cy="7" r="4"></circle>
                            <line x1="20" y1="8" x2="20" y2="14"></line>
                            <line x1="23" y1="11" x2="17" y2="11"></line>
                        </svg>
                        Join Room
                    </button>
                </form>
            </div>
        `,this.setupRoomSelectionEvents()}setupRoomSelectionEvents(){const e=this.shadowRoot.querySelectorAll(".room-tab"),t=this.shadowRoot.querySelector(".password-group"),o=this.shadowRoot.querySelector(".join-password-group"),s=this.shadowRoot.getElementById("connect-btn");e.forEach(i=>{i.addEventListener("click",()=>{e.forEach(r=>r.classList.remove("active")),i.classList.add("active"),i.dataset.mode==="create"?(t.style.display="block",o.style.display="none",s.innerHTML=`
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="8.5" cy="7" r="4"></circle>
                            <line x1="20" y1="8" x2="20" y2="14"></line>
                            <line x1="23" y1="11" x2="17" y2="11"></line>
                        </svg>
                        Create Room
                    `):(t.style.display="none",o.style.display="block",s.innerHTML=`
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="8.5" cy="7" r="4"></circle>
                            <line x1="20" y1="8" x2="20" y2="14"></line>
                            <line x1="23" y1="11" x2="17" y2="11"></line>
                        </svg>
                        Join Room
                    `)})}),s.addEventListener("click",()=>{this.connectToRoom()});const n=this.shadowRoot.getElementById("auto-reconnect-btn");n&&n.addEventListener("click",async()=>{await this.attemptAutoReconnect()||alert("Failed to reconnect automatically. Please try joining manually.")}),document.addEventListener("keypress",i=>{i.key==="Enter"&&this.shadowRoot.getElementById("widget-panel").style.display!=="none"&&this.connectToRoom()})}async connectToRoom(){const e=this.shadowRoot.querySelector(".room-tab.active");if(!e)return;const t=this.shadowRoot.getElementById("username").value.trim(),o=this.shadowRoot.getElementById("room-name").value.trim(),s=e.dataset.mode;if(!t||!o){alert("Please enter both your name and room name");return}this.options.username=t,this.currentRoom=o,this.commentManager&&await this.commentManager.updateCurrentRoom(o);let n=null;s==="create"?n=this.shadowRoot.getElementById("room-password").value.trim()||null:n=this.shadowRoot.getElementById("join-password").value.trim()||null;try{await this.initializeWebRTCChat(),console.log("🎮 Setting webrtcChat reference in gamification component:",this.webrtcChat),this.gamificationComponent.updateWebRTCChat(this.webrtcChat),console.log("🎮 Gamification webrtcChat after assignment:",this.gamificationComponent.webrtcChat),this.quizManager=new C(this),await this.webrtcChat.joinRoom(t,o,n,s),await new Promise(a=>setTimeout(a,1e3)),this.loadPlugins(),await this.initializeCommentSystem(),this.setupGlobalEventListeners(),this.isConnected=!0,this.userCursors.set(this.webrtcChat.myClientId,{persistentId:this.persistentUserId,sessionId:this.webrtcChat.myClientId,username:this.options.username,badges:this.userStats.badges.length>0?this.userStats.badges:this._getRandomBadges(),stars:this.userStats.totalStars,totalScore:this.userStats.totalScore,lastAction:"Joined the room",joinDate:this.userStats.joinDate,totalMessages:this.userStats.totalMessages,totalComments:this.userStats.totalComments}),this.showChatInterface(),this.loadExistingComments(),this.showAIUsageHelp();const i=this.shadowRoot.getElementById("gamification-btn");i&&(i.disabled=!1,i.classList.add("glowing-border")),this.saveRoomSession(t,o,n,s),this.saveChatState()}catch(i){console.error("Failed to connect to room:",i),alert("Failed to connect to room. Please check your connection and try again.")}}showChatInterface(){const e=this.shadowRoot.getElementById("widget-content"),t=e.querySelector(".room-selection");t&&t.remove(),e.innerHTML=`
            <div id="chat-content" class="plugin-content active">
                <div class="user-stats" id="user-stats">
                    <div class="user-stats-header" id="user-stats-header">
                        <span class="user-stats-title" id="user-stats-title">Welcome!</span>
                        <div class="user-stats-controls">
                            <button class="info-btn" id="info-btn" title="Learn about collaborative features">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <circle cx="12" cy="12" r="10"></circle>
                                    <path d="M12 16v-4"></path>
                                    <path d="M12 8h.01"></path>
                                </svg>
                            </button>
                            <button class="user-stats-toggle" id="user-stats-toggle" title="Toggle stats">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <polyline points="6 9 12 15 18 9"></polyline>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="user-stats-content" id="user-stats-content">
                        <!-- User stats will be populated here -->
                    </div>
                </div>
                <div class="chat-container">
                    <div class="chat-messages" id="chat-messages"></div>
                    <div class="typing-indicator" id="typing-indicator" style="display: none;"></div>
                    <div class="chat-input-container">
                        <div class="gemini-input-area">
                            <div class="input-top-row">
                                <textarea class="gemini-textarea" id="chat-input" placeholder="Type a message... (Use @MC for MC assistance)" rows="1"></textarea>
                            </div>
                            <div class="input-bottom-row">
                                <div class="input-left-actions">
                                    <button class="action-btn" id="share-room-btn" title="Share Room">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-4">
                                            <path d="M12 6a2 2 0 1 0-1.994-1.842L5.323 6.5a2 2 0 1 0 0 3l4.683 2.342a2 2 0 1 0 .67-1.342L5.995 8.158a2.03 2.03 0 0 0 0-.316L10.677 5.5c.353.311.816.5 1.323.5Z" />
                                        </svg>
                                    </button>
                                    <button class="action-btn" id="download-chat-btn" title="Download Chat">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-4">
                                            <path d="M8.75 2.75a.75.75 0 0 0-1.5 0v5.69L5.03 6.22a.75.75 0 0 0-1.06 1.06l3.5 3.5a.75.75 0 0 0 1.06 0l3.5-3.5a.75.75 0 0 0-1.06-1.06L8.75 8.44V2.75Z" />
                                            <path d="M3.5 9.75a.75.75 0 0 0-1.5 0v1.5A2.75 2.75 0 0 0 4.75 14h6.5A2.75 2.75 0 0 0 14 11.25v-1.5a.75.75 0 0 0-1.5 0v1.5c0 .69-.56 1.25-1.25 1.25h-6.5c-.69 0-1.25-.56-1.25-1.25v-1.5Z" />
                                        </svg>
                                    </button>
                                </div>
                                <div class="input-right-actions">
                                    <button class="send-btn" id="chat-send" title="Send message">
                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="size-6">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
                                        </svg>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Information Modal -->
            <div class="info-modal" id="info-modal" style="display: none;">
                <div class="info-modal-overlay" id="info-modal-overlay"></div>
                <div class="info-modal-content">
                    <div class="info-modal-header">
                        <h3>Collaborative Learn</h3>
                        <button class="info-modal-close" id="info-modal-close">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                    <div class="info-modal-body">
                        <div class="feature-section">
                            <h4>🚀 Real-time Collaboration</h4>
                            <p>Join rooms with other learners to discuss topics, share insights, and learn together in real-time.</p>
                        </div>
                        
                        <div class="feature-section">
                            <h4>💬 Live Discussions</h4>
                            <p>Engage in live chat discussions with fellow learners. Ask questions, share knowledge, and get instant feedback.</p>
                        </div>
                        
                        <div class="feature-section">
                            <h4>🎯 Live Highlighting</h4>
                            <p>Highlight important content together and see what others are focusing on in real-time.</p>
                        </div>
                        
                        <div class="feature-section">
                            <h4>🤖 AI-Powered MC Quizzes</h4>
                            <p>Take multiple choice quizzes enhanced with AI assistance. Get personalized explanations and learn more effectively.</p>
                        </div>
                        
                        <div class="feature-section">
                            <h4>⭐ Gamification & Points</h4>
                            <p>Earn points for your participation! Active engagement, helpful contributions, and quiz performance all contribute to your learning score.</p>
                        </div>
                        
                        <div class="feature-section">
                            <h4>👥 Social Learning</h4>
                            <p>Learn from peers, share your knowledge, and build a collaborative learning community.</p>
                        </div>
                    </div>
                </div>
            </div>
        `,this.setupTabSwitching(),this.setupChatFunctionality(),this.setupChatActions(),this.setupStatsToggle(),this.setupInfoModal(),this.updateUserStatsDisplayWithAnimation()}setupTabSwitching(){const e=this.shadowRoot.querySelectorAll(".plugin-tab button"),t=this.shadowRoot.querySelectorAll(".plugin-content");e.forEach(o=>{o.addEventListener("click",()=>{const s=o.dataset.plugin;e.forEach(n=>n.classList.remove("active")),o.classList.add("active"),t.forEach(n=>n.classList.remove("active")),this.shadowRoot.getElementById(`${s}-content`).classList.add("active")})})}setupChatFunctionality(){const e=this.shadowRoot.getElementById("chat-input"),t=this.shadowRoot.getElementById("chat-send");if(!e||!t)return;const o=()=>{e.style.height="auto",e.style.height=Math.min(e.scrollHeight,120)+"px"};e.addEventListener("input",o),t.addEventListener("click",()=>{const n=e.value.trim();n&&(this.handleChatMessage(n),e.value="",o())}),e.addEventListener("keydown",n=>{if(n.key==="Enter"&&!n.shiftKey){n.preventDefault();const i=e.value.trim();i&&(this.handleChatMessage(i),e.value="",o())}});let s;e.addEventListener("input",()=>{this.webrtcChat.sendTypingIndicator(),clearTimeout(s),s=setTimeout(()=>{},500)})}setupChatActions(){const e=this.shadowRoot.getElementById("download-chat-btn"),t=this.shadowRoot.getElementById("share-room-btn");e&&e.addEventListener("click",()=>this.downloadChat()),t&&t.addEventListener("click",()=>this.shareRoom())}downloadChat(){const e=this.shadowRoot.getElementById("chat-messages");if(!e)return;const t=Array.from(e.children).map(a=>{const r=a.querySelector(".message-time")?.textContent||"",c=a.querySelector(".message-username")?.textContent||"",l=a.querySelector(".message-content")?.textContent||"",d=a.classList.contains("system-message")?"[SYSTEM]":"[USER]";return`${r} ${d} ${c}: ${l}`}).join(`
`),o=`Chat Export - Room: ${this.currentRoom||"Unknown"}
Date: ${new Date().toLocaleString()}
Total Messages: ${e.children.length}

${t}`,s=new Blob([o],{type:"text/plain"}),n=URL.createObjectURL(s),i=document.createElement("a");i.href=n,i.download=`chat-${this.currentRoom||"export"}-${new Date().toISOString().split("T")[0]}.txt`,document.body.appendChild(i),i.click(),document.body.removeChild(i),URL.revokeObjectURL(n)}shareRoom(){if(this.currentRoomPassword){alert("Cannot share password-protected rooms");return}const e=this.shadowRoot.getElementById("user-count")?.textContent||"0",t=new URL(window.location);t.searchParams.set("room",this.currentRoom),t.searchParams.set("users",e);const o=t.toString();navigator.clipboard.writeText(o).then(()=>{const s=this.shadowRoot.getElementById("share-room-btn");if(s){const n=s.title;s.title="Copied to clipboard!",setTimeout(()=>{s.title=n},2e3)}}).catch(()=>{prompt("Share this room URL:",o)})}setupStatsToggle(){const e=this.shadowRoot.getElementById("user-stats-toggle"),t=this.shadowRoot.getElementById("user-stats-content"),o=this.shadowRoot.getElementById("user-stats-header");if(!e||!t||!o)return;t.classList.remove("expanded"),t.classList.add("collapsed"),e.classList.remove("expanded"),e.classList.add("collapsed");const s=()=>{t.classList.contains("expanded")?(t.classList.remove("expanded"),t.classList.add("collapsed"),e.classList.remove("expanded"),e.classList.add("collapsed")):(t.classList.remove("collapsed"),t.classList.add("expanded"),e.classList.remove("collapsed"),e.classList.add("expanded"))};e.addEventListener("click",n=>{n.stopPropagation(),s()}),o.addEventListener("click",s)}setupInfoModal(){const e=this.shadowRoot.getElementById("info-btn"),t=this.shadowRoot.getElementById("info-modal"),o=this.shadowRoot.getElementById("info-modal-close"),s=this.shadowRoot.getElementById("info-modal-overlay");!e||!t||!o||!s||(e.addEventListener("click",n=>{n.stopPropagation(),t.style.display="flex",setTimeout(()=>{t.classList.add("show")},10)}),o.addEventListener("click",()=>{t.classList.remove("show"),setTimeout(()=>{t.style.display="none"},200)}),s.addEventListener("click",()=>{t.classList.remove("show"),setTimeout(()=>{t.style.display="none"},200)}),document.addEventListener("keydown",n=>{n.key==="Escape"&&t.style.display==="flex"&&(t.classList.remove("show"),setTimeout(()=>{t.style.display="none"},200))}))}setupWidgetControls(){const e=this.shadowRoot.getElementById("widget-toggle"),t=this.shadowRoot.getElementById("widget-panel"),o=this.shadowRoot.getElementById("minimize-btn"),s=this.shadowRoot.getElementById("exit-btn"),n=this.shadowRoot.getElementById("gamification-btn");n.disabled=!0,e.addEventListener("click",()=>{t.style.display==="none"?(t.style.display="block",e.style.display="none",this.isMinimized=!1):(t.style.display="none",e.style.display="flex",this.isMinimized=!0),this.saveChatState()}),o.addEventListener("click",()=>{t.style.display="none",e.style.display="flex",this.isMinimized=!0,this.saveChatState()}),s.addEventListener("click",()=>{this.webrtcChat&&this.webrtcChat.disconnect(),this.clearRoomSession(),this.showRoomSelection();const i=this.shadowRoot.getElementById("gamification-btn");i&&(i.disabled=!0,i.classList.remove("glowing-border")),this.gamificationComponent.hide()}),n.addEventListener("click",()=>{this.gamificationComponent.toggle(),this.gamificationComponent.isVisible?n.classList.remove("glowing-border"):n.classList.add("glowing-border")})}async initializeWebRTCChat(){this.webrtcChat=new w({signalingServerUrl:this.options.signalingServerUrl,username:this.options.username,onConnectionStatusChange:this.onConnectionStatusChange.bind(this),onUserCountChange:this.onUserCountChange.bind(this),onMessage:this.onMessage.bind(this),onInfoMessage:this.onInfoMessage.bind(this),onTypingIndicatorChange:this.onTypingIndicatorChange.bind(this),onUserJoined:this.onUserJoined.bind(this),onUserLeft:this.onUserLeft.bind(this),onCommentAdd:this.onCommentAdd.bind(this),onCommentDelete:this.onCommentDelete.bind(this),onCommentResolve:this.onCommentResolve.bind(this),onCommentStatusRequest:this.onCommentStatusRequest.bind(this),onCommentStatusResponse:this.onCommentStatusResponse.bind(this),onSystemMessage:this.onSystemMessage.bind(this),onCursorUpdate:this.handleCursorUpdate.bind(this),onPresenceUpdate:this.handlePresenceUpdate.bind(this)})}async initializeCommentSystem(){try{if(this.commentManager&&this.commentManager.isInitialized){console.log("📝 Comment system already initialized");return}if(!this.currentRoom||!this.webrtcChat){console.warn("⚠️ Missing dependencies for comment system initialization:",{currentRoom:!!this.currentRoom,webrtcChat:!!this.webrtcChat,userCursors:!!this.userCursors});return}this.userCursors||(this.userCursors=new Map,console.log("📝 Initialized userCursors for comment system")),this.commentManager=new K({currentRoom:this.currentRoom,webrtcChat:this.webrtcChat,userCursors:this.userCursors}),await this.commentManager.initialize(),this.commentDB=this.commentManager.commentDB,console.log("📝 Comment system initialized for room:",this.currentRoom)}catch(e){console.error("❌ Failed to initialize comment system:",e)}}loadPlugins(){this.options.enablePresence&&this.initializePresence()}initializePresence(){this.userCursors=new Map,this.cursorElements=new Map,this.colors=["#ff4757","#2ed573","#1e90ff","#ffa502","#ff6348","#7bed9f","#70a1ff","#ff7675"],this.colorIndex=0,setInterval(()=>{this.sendPresenceUpdate()},2e3),setInterval(()=>{this.updatePresenceList()},1e3)}sendPresenceUpdate(){if(this.webrtcChat){const e=this.userCursors.get(this.webrtcChat.myClientId)||{};this.webrtcChat.broadcastMessage({type:"presence_update",status:"active",page:window.location.href,timestamp:Date.now(),persistentId:this.persistentUserId,sessionId:this.webrtcChat.myClientId,username:this.options.username,stars:e.stars||0,totalScore:e.totalScore||0,badges:e.badges||[],joinDate:e.joinDate||this.userStats.joinDate,totalMessages:e.totalMessages||0,totalComments:e.totalComments||0})}}_getRandomBadges(){const e=[];return Math.random()>.5&&e.push({icon:"💬",text:"Chatterbox"}),Math.random()>.7&&e.push({icon:"✍️",text:"Commentator"}),e}updatePresenceList(){const e=this.shadowRoot.getElementById("presence-list"),t=[],o=this.webrtcChat.myClientId;if(o&&this.webrtcChat.myUsername){const s=this.userCursors.get(o)||{};t.push({peerId:o,persistentId:s.persistentId||this.persistentUserId,sessionId:s.sessionId||o,username:this.webrtcChat.myUsername,page:window.location.href,stars:s.stars||0,totalScore:s.totalScore||0,lastAction:s.lastAction||"",badges:s.badges||[],joinDate:s.joinDate||this.userStats.joinDate,totalMessages:s.totalMessages||0,totalComments:s.totalComments||0})}if(this.userCursors.forEach((s,n)=>{n!==o&&t.push({peerId:n,persistentId:s.persistentId||null,sessionId:s.sessionId||n,username:s.username,page:s.page,stars:s.stars||0,totalScore:s.totalScore||0,lastAction:s.lastAction||"",badges:s.badges||[],joinDate:s.joinDate||Date.now(),totalMessages:s.totalMessages||0,totalComments:s.totalComments||0})}),e){const s=t.map(n=>n.username);s.length<=1?e.textContent="Only you are active":e.innerHTML=s.map(n=>`<div style="margin-bottom: 4px;">• ${n}</div>`).join("")}this.gamificationComponent&&(t.forEach(s=>{s.stars!==s.totalScore&&(console.warn(`Aligning stars/score for ${s.username}: stars=${s.stars}, totalScore=${s.totalScore}`),s.stars=s.totalScore)}),this.gamificationComponent.update(t)),this.updateUserStatsDisplayIfChanged()}handlePresenceUpdate(e,t){const o=this.webrtcChat.peers[t]?.username||`User_${t}`;this.userCursors||(this.userCursors=new Map);let s=this.userCursors.get(t);s||(s={persistentId:null,sessionId:t,username:o,badges:this._getRandomBadges(),stars:0,totalScore:0,lastAction:"Joined the room",joinDate:Date.now(),totalMessages:0,totalComments:0}),s.page=e.page,s.timestamp=e.timestamp,s.color=this.getUserColor(t),e.persistentId&&(s.persistentId=e.persistentId),e.sessionId&&(s.sessionId=e.sessionId),e.stars!==void 0&&(s.stars=e.stars),e.totalScore!==void 0&&(s.totalScore=e.totalScore),e.badges&&(s.badges=e.badges),e.joinDate&&(s.joinDate=e.joinDate),e.totalMessages!==void 0&&(s.totalMessages=e.totalMessages),e.totalComments!==void 0&&(s.totalComments=e.totalComments),this.userCursors.set(t,s),this.updatePresenceList()}handleCursorUpdate(e,t){const o=this.webrtcChat.peers[t]?.username||`User_${t}`,s=this.userCursors.get(t)||{};this.userCursors.set(t,{...s,username:o,x:e.x,y:e.y,timestamp:e.timestamp,color:this.getUserColor(t)}),this.updateCursorDisplay(t)}getUserColor(e){if(!this.userCursors.has(e)){(!this.colors||!this.colors.length)&&(this.colors=["#ff4757","#2ed573","#1e90ff","#ffa502","#ff6348","#7bed9f","#70a1ff","#ff7675"],this.colorIndex=0);const t=this.colors[this.colorIndex%this.colors.length];return this.colorIndex++,t}return this.userCursors.get(e).color}updateCursorDisplay(e){const t=this.userCursors.get(e);if(!t)return;let o=this.cursorElements.get(e);o||(o=document.createElement("div"),o.className="presence-indicator",document.body.appendChild(o),this.cursorElements.set(e,o)),o.timeout&&clearTimeout(o.timeout);const s=this.shadowRoot.querySelector(".widget-container"),n=getComputedStyle(s),i=n.getPropertyValue("--presence-label-background-color").trim(),a=n.getPropertyValue("--presence-label-text-color").trim(),r=n.getPropertyValue("--presence-label-border-color").trim();o.innerHTML=`
            <div class="presence-cursor" style="border-color: ${t.color};"></div>
            <div class="presence-label" style="background-color: ${i}; color: ${a}; border-color: ${r};">${t.username}</div>
        `,o.style.left=t.x+"px",o.style.top=t.y+"px",o.timeout=setTimeout(()=>{this.cursorElements.has(e)&&(this.cursorElements.get(e).remove(),this.cursorElements.delete(e),this.userCursors.delete(e))},5e3)}injectGlobalStyles(){const e=document.createElement("style");e.textContent=`
            .presence-indicator {
                position: absolute;
                pointer-events: none;
                z-index: 9999;
                transition: all 0.1s ease;
            }
            .presence-cursor {
                width: 20px;
                height: 20px;
                border: 2px solid;
                border-radius: 50%;
                position: relative;
            }
            .presence-label {
                position: absolute;
                top: 25px;
                left: 0;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 11px;
                font-family: "Tinos", serif;
                font-weight: 700;
                white-space: nowrap;
                pointer-events: none;
                border: 1px solid;
            }
            .remote-highlight {
                background-color: yellow; /* Default color */
                transition: background-color 0.3s ease;
                border-radius: 3px;
            }
        `,document.head.appendChild(e)}injectMarkdownStyles(){if(this.shadowRoot&&this.shadowRoot.querySelector("#markdown-styles")||document.querySelector("#markdown-styles"))return;const e=document.createElement("style");e.id="markdown-styles",e.textContent=`
            /* AI Message Rainbow Gradient Background */
            .chat-message.ai-message {
                background: #C9D6FF;  /* fallback for old browsers */
                background: -webkit-linear-gradient(to right, #E2E2E2, #C9D6FF);  /* Chrome 10-25, Safari 5.1-6 */
                background: linear-gradient(to right, #E2E2E2, #C9D6FF); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
                border-radius: 16px;
                box-shadow: 0 4px 16px rgba(255, 105, 180, 0.2), 
                           0 0 20px rgba(138, 43, 226, 0.1);
                position: relative;
                overflow: hidden;
                max-width: 95% !important;
                width: auto;
            }
            
            
            .chat-message.ai-message::after {
                content: '';
                position: absolute;
                top: -2px;
                left: -2px;
                right: -2px;
                bottom: -2px;
                background: linear-gradient(45deg, 
                    rgba(255, 105, 180, 0.3),
                    rgba(255, 165, 0, 0.3),
                    rgba(255, 255, 0, 0.3),
                    rgba(0, 255, 127, 0.3),
                    rgba(0, 191, 255, 0.3),
                    rgba(138, 43, 226, 0.3),
                    rgba(255, 20, 147, 0.3)
                );
                border-radius: 18px;
                z-index: -1;
                animation: rainbow-border 6s infinite linear;
            }
            
            
            @keyframes rainbow-border {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .chat-message.ai-message .username {
                background: linear-gradient(45deg, #ff69b4, #8b5cf6, #00bfff, #ff1493);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 700;
                margin-bottom: 8px;
            }
            
            /* Markdown Styles for AI Messages */
            .message-text h1 {
                font-size: 1.5em;
                font-weight: bold;
                color: #2563eb;
                margin: 0.5em 0;
                border-bottom: 2px solid #e5e7eb;
                padding-bottom: 0.25em;
            }
            
            .message-text h2 {
                font-size: 1.25em;
                font-weight: bold;
                color: #2563eb;
                margin: 0.4em 0;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 0.2em;
            }
            
            .message-text h3 {
                font-size: 1.1em;
                font-weight: bold;
                color: #2563eb;
                margin: 0.3em 0;
            }
            
            .message-text strong {
                font-weight: bold;
            }
            
            .message-text em {
                font-style: italic;
            }
            
            .message-text code {
                background-color: #f3f4f6;
                padding: 0.125em 0.25em;
                border-radius: 0.25em;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.875em;
            }
            
            .message-text pre {
                background-color: #f3f4f6;
                padding: 1em;
                border-radius: 0.5em;
                overflow-x: auto;
                margin: 0.5em 0;
            }
            
            .message-text pre code {
                background: none;
                padding: 0;
                border-radius: 0;
            }
            
            .message-text a {
                color: #2563eb;
                text-decoration: underline;
            }
            
            .message-text a:hover {
                color: #1d4ed8;
            }
            
            .message-text ul, .message-text ol {
                margin: 0.5em 0;
                padding-left: 1.5em;
            }
            
            .message-text li {
                margin: 0.25em 0;
            }
            
            .message-text p {
                margin: 1em 0;
            }
            
            .message-text p:first-child {
                margin-top: 0.5em;
            }
            
            .message-text p:last-child {
                margin-bottom: 0.5em;
            }
            
            .message-text blockquote {
                border-left: 4px solid #e5e7eb;
                padding-left: 1em;
                margin: 0.5em 0;
                color: #6b7280;
            }
        `,this.shadowRoot?(this.shadowRoot.appendChild(e),console.log("Markdown styles injected into shadow DOM")):(document.head.appendChild(e),console.log("Markdown styles injected into document head (fallback)"))}injectComponentStyles(){if(this.shadowRoot&&this.shadowRoot.querySelector("#component-styles")||document.querySelector("#component-styles"))return;const e=document.createElement("style");e.id="component-styles",e.textContent=`
            /* Message Component Styles */

            /* Quiz Component Styles */
            .message-quiz-component {
              background: linear-gradient(135deg, 
                rgba(255, 105, 180, 0.1) 0%,
                rgba(138, 43, 226, 0.1) 50%,
                rgba(0, 191, 255, 0.1) 100%
              );
              border: 2px solid rgba(255, 105, 180, 0.3);
              border-radius: 12px;
              padding: 16px;
              margin: 12px 0;
              position: relative;
              overflow: hidden;
            }

            .message-quiz-component::before {
              content: '';
              position: absolute;
              top: -2px;
              left: -2px;
              right: -2px;
              bottom: -2px;
              background: linear-gradient(45deg, 
                rgba(255, 105, 180, 0.2),
                rgba(255, 165, 0, 0.2),
                rgba(255, 255, 0, 0.2),
                rgba(0, 255, 127, 0.2),
                rgba(0, 191, 255, 0.2),
                rgba(138, 43, 226, 0.2),
                rgba(255, 20, 147, 0.2)
              );
              border-radius: 14px;
              z-index: -1;
              animation: rainbow-border 6s infinite linear;
            }

            @keyframes rainbow-border {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }

            .quiz-header {
              display: flex;
              align-items: center;
              gap: 8px;
              margin-bottom: 12px;
            }

            .quiz-icon {
              font-size: 20px;
            }

            .quiz-title {
              font-weight: 600;
              color: #8b5cf6;
              font-size: 16px;
            }

            .quiz-question {
              margin-bottom: 16px;
            }

            .quiz-question p {
              margin: 0;
              font-size: 16px;
              line-height: 1.5;
              color: #374151;
            }

            .quiz-choices {
              display: flex;
              flex-direction: column;
              gap: 8px;
              margin-bottom: 16px;
            }

            .quiz-submit-container {
              display: flex;
              justify-content: center;
              margin-bottom: 16px;
            }

            .quiz-submit-btn {
              display: flex;
              align-items: center;
              gap: 8px;
              padding: 12px 24px;
              background: linear-gradient(135deg, #8b5cf6, #3b82f6);
              color: white;
              border: none;
              border-radius: 8px;
              font-size: 14px;
              font-weight: 600;
              cursor: pointer;
              transition: all 0.2s ease;
              box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
            }

            .quiz-submit-btn:hover:not(:disabled) {
              background: linear-gradient(135deg, #7c3aed, #2563eb);
              transform: translateY(-1px);
              box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
            }

            .quiz-submit-btn:disabled {
              background: #9ca3af;
              cursor: not-allowed;
              transform: none;
              box-shadow: none;
            }

            .submit-icon {
              font-size: 16px;
            }

            .submit-text {
              font-size: 14px;
            }

            .quiz-choice {
              display: flex;
              align-items: center;
              gap: 12px;
              padding: 12px 16px;
              background: rgba(255, 255, 255, 0.8);
              border: 2px solid rgba(255, 105, 180, 0.2);
              border-radius: 8px;
              cursor: pointer;
              transition: all 0.2s ease;
              font-size: 14px;
              position: relative;
            }

            .quiz-choice:hover {
              background: rgba(255, 105, 180, 0.1);
              border-color: rgba(255, 105, 180, 0.4);
            }

            .quiz-radio {
              position: absolute;
              opacity: 0;
              pointer-events: none;
              margin: 0;
              padding: 0;
            }

            .quiz-choice:has(.quiz-radio:checked) {
              background: rgba(255, 105, 180, 0.2);
              border-color: rgba(255, 105, 180, 0.6);
              box-shadow: 0 2px 8px rgba(255, 105, 180, 0.2);
            }

            .quiz-choice:has(.quiz-radio:checked) .choice-letter {
              background: rgba(255, 105, 180, 0.4);
              color: white;
            }

            .quiz-choice.correct {
              background: rgba(34, 197, 94, 0.2);
              border-color: rgba(34, 197, 94, 0.6);
            }

            .quiz-choice.incorrect {
              background: rgba(239, 68, 68, 0.2);
              border-color: rgba(239, 68, 68, 0.6);
            }

            .choice-feedback {
              margin-top: 6px;
              margin-bottom: 8px;
              margin-left: 20px;
              padding: 6px 10px;
              border-radius: 6px;
              font-size: 13px;
              font-weight: 500;
              display: block;
            }

            .choice-feedback.correct {
              background: rgba(34, 197, 94, 0.1);
              color: #059669;
              border: 1px solid rgba(34, 197, 94, 0.3);
            }

            .choice-feedback.incorrect {
              background: rgba(239, 68, 68, 0.1);
              color: #dc2626;
              border: 1px solid rgba(239, 68, 68, 0.3);
            }

            /* Removed quiz-feedback-container styles since we're inserting feedback directly after choices */

            .choice-letter {
              display: flex;
              align-items: center;
              justify-content: center;
              width: 24px;
              height: 24px;
              background: rgba(255, 105, 180, 0.2);
              border-radius: 50%;
              font-weight: 600;
              font-size: 12px;
              color: #8b5cf6;
            }

            .choice-text {
              flex: 1;
              color: #374151;
            }

            .quiz-explanation {
              margin-top: 16px;
              padding: 16px;
              background: rgba(255, 255, 255, 0.5);
              border-radius: 8px;
              border: 1px solid rgba(255, 105, 180, 0.2);
            }

            .explanation-header {
              display: flex;
              align-items: center;
              gap: 8px;
              margin-bottom: 12px;
            }

            .explanation-icon {
              font-size: 16px;
            }

            .explanations-content {
              display: flex;
              flex-direction: column;
              gap: 8px;
            }

            .explanation-item {
              display: flex;
              align-items: flex-start;
              gap: 12px;
              padding: 8px 12px;
              border-radius: 6px;
            }

            .explanation-item.correct {
              background: rgba(34, 197, 94, 0.1);
              border-left: 4px solid #22c55e;
            }

            .explanation-item.incorrect {
              background: rgba(239, 68, 68, 0.1);
              border-left: 4px solid #ef4444;
            }

            .explanation-letter {
              display: flex;
              align-items: center;
              justify-content: center;
              width: 20px;
              height: 20px;
              background: rgba(255, 105, 180, 0.2);
              border-radius: 50%;
              font-weight: 600;
              font-size: 11px;
              color: #8b5cf6;
              flex-shrink: 0;
            }

            .explanation-text {
              flex: 1;
              font-size: 14px;
              line-height: 1.4;
              color: #374151;
            }

            .quiz-result {
              display: flex;
              align-items: center;
              gap: 8px;
              margin-top: 16px;
              padding: 12px 16px;
              background: rgba(255, 255, 255, 0.8);
              border-radius: 8px;
              border: 2px solid rgba(255, 105, 180, 0.3);
            }

            .result-icon {
              font-size: 18px;
            }

            .result-text {
              font-weight: 600;
              color: #374151;
            }

            /* Quiz Kapoot Component Styles */
            .quiz-kapoot-active {
              background: linear-gradient(135deg, 
                rgba(255, 105, 180, 0.1) 0%,
                rgba(138, 43, 226, 0.1) 50%,
                rgba(0, 191, 255, 0.1) 100%
              );
              border: 2px solid rgba(255, 105, 180, 0.3);
              border-radius: 12px;
              padding: 16px;
              margin: 12px 0;
              position: relative;
              overflow: hidden;
            }

            .quiz-kapoot-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 16px;
            }

            .quiz-kapoot-title {
              font-size: 18px;
              font-weight: bold;
              color: #1f2937 !important;
            }

            .quiz-kapoot-timer {
              font-size: 16px;
              font-weight: bold;
              padding: 8px 12px;
              background: rgba(255, 255, 255, 0.8) !important;
              border-radius: 8px;
              color: #1f2937 !important;
            }

            .quiz-kapoot-timer.safe { color: #4CAF50; }
            .quiz-kapoot-timer.warning { color: #FF9800; }
            .quiz-kapoot-timer.critical { color: #F44336; }

            .quiz-kapoot-question-info {
              display: flex;
              justify-content: space-between;
              margin-bottom: 12px;
              font-size: 14px;
              color: #555;
            }

            .quiz-kapoot-question {
              font-size: 16px;
              font-weight: bold;
              color: #333;
              margin-bottom: 16px;
              line-height: 1.4;
            }

            .quiz-kapoot-choices {
              display: grid;
              gap: 8px;
              margin-bottom: 16px;
            }

            .quiz-kapoot-choice {
              display: flex;
              align-items: center;
              padding: 12px;
              background: rgba(255, 255, 255, 0.8) !important;
              border: 2px solid rgba(255, 105, 180, 0.3);
              border-radius: 8px;
              cursor: pointer;
              transition: all 0.2s ease;
              color: #1f2937 !important;
            }

            .quiz-kapoot-choice:hover {
              background: rgba(255, 105, 180, 0.1) !important;
              border-color: rgba(255, 105, 180, 0.5);
              transform: translateY(-2px);
            }

            .quiz-choice-letter {
              font-weight: bold;
              margin-right: 12px;
              min-width: 24px;
              text-align: center;
            }

            .quiz-choice-text {
              flex: 1;
            }

            .quiz-kapoot-team-status {
              margin-bottom: 12px;
              font-size: 14px;
              color: #555;
            }

            .quiz-kapoot-sources {
              margin-bottom: 12px;
              font-size: 14px;
              color: #6b7280 !important;
            }

            .quiz-kapoot-sources a {
              color: #4CAF50;
              text-decoration: none;
            }

            .quiz-kapoot-sources a:hover {
              text-decoration: underline;
            }

            .quiz-kapoot-note {
              font-size: 12px;
              color: #9ca3af !important;
              font-style: italic;
            }

            /* JSON Component Styles */
            .message-json-component {
              background: rgba(248, 250, 252, 0.8);
              border: 1px solid rgba(203, 213, 225, 0.5);
              border-radius: 8px;
              padding: 16px;
              margin: 12px 0;
            }

            .json-header {
              display: flex;
              align-items: center;
              gap: 8px;
              margin-bottom: 12px;
            }

            .json-icon {
              font-size: 16px;
            }

            .json-title {
              font-weight: 600;
              color: #64748b;
              font-size: 14px;
            }

            .json-content {
              background: #f8fafc;
              border: 1px solid #e2e8f0;
              border-radius: 6px;
              padding: 12px;
              overflow-x: auto;
            }

            .json-content pre {
              margin: 0;
              font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
              font-size: 12px;
              line-height: 1.4;
              color: #334155;
            }

            .json-content code {
              background: none;
              padding: 0;
              border-radius: 0;
            }

            /* Component Error Styles */
            .component-error {
              background: rgba(239, 68, 68, 0.1);
              border: 1px solid rgba(239, 68, 68, 0.3);
              border-radius: 6px;
              padding: 12px;
              margin: 12px 0;
              color: #dc2626;
              font-size: 14px;
              text-align: center;
            }
        `,this.shadowRoot?(this.shadowRoot.appendChild(e),console.log("Component styles injected into shadow DOM")):(document.head.appendChild(e),console.log("Component styles injected into document head (fallback)"))}updateCursor(e,t){this.webrtcChat&&this.webrtcChat.broadcastMessage({type:"cursor_update",x:e,y:t,timestamp:Date.now()})}loadPlugin(e,t){this.plugins.set(e,t),t.initialize(this.widgetContainer)}setupGlobalEventListeners(){this.options.enablePresence&&document.addEventListener("mousemove",e=>{this.updateCursor(e.clientX,e.clientY)}),this.setupSelectionToolbar(),this.setupNavigationListeners()}setupNavigationListeners(){window.addEventListener("beforeunload",()=>{}),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&this.isConnected&&this.sendPresenceUpdate()}),window.addEventListener("popstate",()=>{this.isConnected&&setTimeout(()=>{this.sendPresenceUpdate()},500)}),window.addEventListener("hashchange",()=>{this.isConnected&&setTimeout(()=>{this.sendPresenceUpdate()},500)})}getSelectorForNode(e){if(e.nodeType!==1&&(e=e.parentNode),e.id)return`#${e.id}`;if(e.tagName==="BODY")return"BODY";let t=[];for(;e.parentElement&&e.tagName!=="BODY";){let o=e.tagName.toLowerCase(),s=e,n=1;for(;s=s.previousElementSibling;)s.tagName===e.tagName&&n++;o+=`:nth-of-type(${n})`,t.unshift(o),e=e.parentElement}return`BODY > ${t.join(" > ")}`}getTextNode(e,t){if(e.nodeType===Node.TEXT_NODE)return{node:e,offset:t};let o=0;for(let s of e.childNodes)if(s.nodeType===Node.TEXT_NODE){const n=s.textContent.length;if(o+n>=t)return{node:s,offset:t-o};o+=n}else if(s.nodeType===Node.ELEMENT_NODE){const n=this.getTextNode(s,t-o);if(n)return n;o+=s.textContent.length}return null}setupSelectionToolbar(){let e,t=!1;document.addEventListener("mousedown",()=>{t=!0}),document.addEventListener("mouseup",()=>{t=!1,setTimeout(()=>{t||this.checkForSelection()},50)}),document.addEventListener("selectionchange",()=>{t||(clearTimeout(e),e=setTimeout(()=>{this.checkForSelection()},100))}),this.checkForSelection=()=>{const o=document.getSelection();if(this.currentToolbar&&(this.currentToolbar.remove(),this.currentToolbar=null),!o||o.rangeCount===0||o.isCollapsed||!this.currentRoom)return;const s=o.getRangeAt(0),n=o.toString().trim();n.length<3||(s.cloneRange(),Date.now(),this.showSelectionToolbar(s,n))},document.addEventListener("click",o=>{this.currentToolbar&&!this.currentToolbar.contains(o.target)&&setTimeout(()=>{this.currentToolbar&&!this.currentToolbar.contains(o.target)&&(this.currentToolbar.remove(),this.currentToolbar=null,this.preservedSelection=null)},200)}),document.addEventListener("keydown",o=>{this.currentToolbar&&o.key==="Escape"&&(this.currentToolbar.remove(),this.currentToolbar=null,this.preservedSelection=null)})}showSelectionToolbar(e,t){const o=document.querySelector(".selection-toolbar");o&&o.remove(),this.preservedSelection={range:e.cloneRange(),text:t,timestamp:Date.now()};const s=document.createElement("div");s.className="selection-toolbar",s.style.cssText=`
            position: absolute;
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            display: flex;
            gap: 8px;
            align-items: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.2s ease;
        `,setTimeout(()=>{s.style.opacity="1"},10);const n=document.createElement("button");n.innerHTML="💬 Comment",n.style.cssText=`
            background: #007bff;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: background 0.2s ease;
        `,n.addEventListener("mouseenter",()=>{n.style.background="#0056b3"}),n.addEventListener("mouseleave",()=>{n.style.background="#007bff"}),n.addEventListener("mousedown",r=>{r.preventDefault(),r.stopPropagation()}),n.addEventListener("click",r=>{r.preventDefault(),r.stopPropagation();const c=this.preservedSelection||{range:e,text:t};this.createCommentFromSelection(c.range,c.text,r),s.remove(),this.preservedSelection=null}),s.appendChild(n);const i=e.getBoundingClientRect();s.style.left=`${i.left+window.scrollX}px`,s.style.top=`${i.bottom+window.scrollY+5}px`;const a=s.getBoundingClientRect();a.right>window.innerWidth&&(s.style.left=`${window.innerWidth-a.width-10}px`),a.bottom>window.innerHeight&&(s.style.top=`${i.top+window.scrollY-a.height-5}px`),document.body.appendChild(s),this.currentToolbar=s,this.restoreSelection=()=>{if(this.preservedSelection)try{const r=document.getSelection();r.removeAllRanges(),r.addRange(this.preservedSelection.range)}catch(r){console.warn("Could not restore selection:",r)}},s.addEventListener("mouseenter",()=>{this.restoreSelection()})}async createCommentFromSelection(e,t,o){try{if((!this.commentManager||!this.commentManager.isInitialized)&&(console.log("📝 Attempting to initialize comment system..."),await this.initializeCommentSystem()),this.commentManager&&this.commentManager.isInitialized){const s={url:window.location.href,startSelector:this.getSelectorForNode(e.startContainer),startOffset:e.startOffset,endSelector:this.getSelectorForNode(e.endContainer),endOffset:e.endOffset,color:this.getUserColor(this.webrtcChat.myClientId),selectedText:t};console.log("📝 Creating comment with selection data:",s),await this.commentManager.createCommentFromSelection(s)}else console.warn("⚠️ Comment manager not available - falling back to legacy comment system"),this.showCommentPanel({url:window.location.href,startSelector:this.getSelectorForNode(e.startContainer),startOffset:e.startOffset,endSelector:this.getSelectorForNode(e.endContainer),endOffset:e.endOffset,color:this.getUserColor(this.webrtcChat.myClientId),selectedText:t},o)}catch(s){console.error("Failed to create comment from selection:",s)}}showCommentPanel(e,t){const o=document.querySelector(".comment-tooltip");o&&o.remove();const s=document.createElement("div");s.className="comment-tooltip",s.style.cssText=`
            position: fixed;
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10001;
            width: 250px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;const n=document.createElement("div");n.style.cssText=`
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 6px 8px;
            margin-bottom: 10px;
            font-size: 11px;
            color: #666;
            max-height: 40px;
            overflow-y: auto;
        `,n.textContent=`"${e.selectedText}"`;const i=document.createElement("textarea");i.placeholder="Add a comment...",i.style.cssText=`
            width: 100%;
            height: 50px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 6px;
            font-size: 13px;
            resize: none;
            margin-bottom: 8px;
            font-family: inherit;
            box-sizing: border-box;
        `;const a=document.createElement("div");a.style.cssText="display: flex; gap: 6px; justify-content: flex-end;";const r=document.createElement("button");r.textContent="Save",r.style.cssText=`
            background: #007bff;
            color: white;
            border: none;
            padding: 4px 12px;
            border-radius: 3px;
            font-size: 12px;
            cursor: pointer;
            font-weight: 500;
        `;const c=document.createElement("button");c.textContent="Cancel",c.style.cssText=`
            background: #6c757d;
            color: white;
            border: none;
            padding: 4px 12px;
            border-radius: 3px;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.8;
        `,r.addEventListener("click",()=>{const u=i.value.trim();u&&this.saveComment(u,this.webrtcChat.myClientId,e),s.remove()}),c.addEventListener("click",()=>{s.remove()}),a.appendChild(c),a.appendChild(r),s.appendChild(n),s.appendChild(i),s.appendChild(a);const l=t&&t.clientX||window.innerWidth/2,d=t&&t.clientY||window.innerHeight/2;s.style.left=`${l-125}px`,s.style.top=`${d-10}px`;const h=s.getBoundingClientRect();h.right>window.innerWidth&&(s.style.left=`${window.innerWidth-h.width-10}px`),h.bottom>window.innerHeight&&(s.style.top=`${window.innerHeight-h.height-10}px`),document.body.appendChild(s),i.focus();const m=u=>{s.contains(u.target)||(s.remove(),document.removeEventListener("click",m))};setTimeout(()=>document.addEventListener("click",m),0)}async createHighlightFromComment(e){try{const t=e.highlightData;if(document.getElementById(e.highlightId))return;const s=document.querySelector(t.startSelector),n=document.querySelector(t.endSelector);if(!s||!n){console.warn("Could not create highlight for comment:",e.id);return}const i=this.getTextNode(s,t.startOffset),a=this.getTextNode(n,t.endOffset);if(!i||!a){console.warn("Could not find text nodes for highlight:",e.id);return}const r=document.createRange();r.setStart(i.node,i.offset),r.setEnd(a.node,a.offset);const c=document.createElement("span");c.className="remote-highlight",c.id=e.highlightId;const l=this.getUserColor(e.peerId);c.style.backgroundColor=l,c.style.mixBlendMode="multiply",c.style.position="relative",c.appendChild(r.extractContents()),r.insertNode(c),this.addHighlightHoverTooltip(c,e)}catch(t){console.error("Failed to create highlight from comment:",t)}}addHighlightHoverTooltip(e,t){let o=null,s=null,n=null;const i=()=>{o&&(o.remove(),o=null),n&&(clearTimeout(n),n=null)},a=()=>{o&&o.remove(),o=document.createElement("div"),o.className="highlight-tooltip",o.style.cssText=`
                position: absolute;
                background: white;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 8px 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 10000;
                max-width: 200px;
                font-size: 12px;
                line-height: 1.3;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            `;const r=document.createElement("div");r.style.cssText=`
                color: #333;
                margin-bottom: 4px;
                word-wrap: break-word;
            `,r.textContent=t.text;const c=document.createElement("div");c.style.cssText=`
                font-size: 10px;
                color: #666;
                border-top: 1px solid #eee;
                padding-top: 4px;
                margin-bottom: 6px;
            `,c.textContent=`${t.author} • ${new Date(t.timestamp).toLocaleTimeString()}`;const l=document.createElement("button");l.textContent="Resolve",l.style.cssText=`
                background: #6c757d;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 10px;
                cursor: pointer;
                margin-top: 5px;
                width: 100%;
                font-weight: 500;
                transition: background 0.2s ease;
                opacity: 0.8;
            `,l.addEventListener("mouseenter",()=>{l.style.background="#5a6268",l.style.opacity="1"}),l.addEventListener("mouseleave",()=>{l.style.background="#6c757d",l.style.opacity="0.8"}),l.addEventListener("click",()=>{this.resolveComment(t.id,t.highlightId),i()}),o.appendChild(r),o.appendChild(c),o.appendChild(l);const d=e.getBoundingClientRect();o.style.left=`${d.left+window.scrollX}px`,o.style.top=`${d.bottom+window.scrollY+5}px`;const h=o.getBoundingClientRect();h.right>window.innerWidth&&(o.style.left=`${window.innerWidth-h.width-10}px`),h.bottom>window.innerHeight&&(o.style.top=`${d.top+window.scrollY-h.height-5}px`),document.body.appendChild(o),o.addEventListener("mouseenter",()=>{n&&(clearTimeout(n),n=null)}),o.addEventListener("mouseleave",()=>{n=setTimeout(()=>{i()},200)})};e.addEventListener("mouseenter",()=>{s&&clearTimeout(s),n&&(clearTimeout(n),n=null),s=setTimeout(()=>{a()},300)}),e.addEventListener("mouseleave",()=>{s&&(clearTimeout(s),s=null),n=setTimeout(()=>{i()},200)})}async saveComment(e,t,o){try{const s={id:`${t}-${Date.now()}-${Math.random().toString(36).substr(2,9)}`,highlightId:`highlight-${t}-${Date.now()}-${Math.random().toString(36).substr(2,9)}`,peerId:t,text:e,url:o.url,room:this.currentRoom,highlightData:o,timestamp:Date.now(),author:this.webrtcChat.myClientId===t?"You":`User ${t.substring(0,8)}`};if(this.commentDB&&await this.commentDB.saveComment(s),await this.createHighlightFromComment(s),this.webrtcChat){this.webrtcChat.broadcastMessage({type:"comment_add",data:s});const i=this.webrtcChat.myUsername||"You",a=document.title||"this page",r={message:`💬 ${i} added a comment on "${a}"`,url:window.location.href,commentId:s.id,author:i,pageTitle:a};this.webrtcChat.broadcastMessage({type:"system_message",payload:r}),this.onSystemMessage(r,this.webrtcChat.myClientId)}this.hasComments.add(s.highlightId);const n=this.userCursors.get(t)||{};t===this.webrtcChat.myClientId?(this.incrementUserAction("comment"),this.userCursors.set(t,{...n,lastAction:"Added a comment",stars:this.userStats.totalStars,totalScore:this.userStats.totalScore,totalComments:this.userStats.totalComments})):this.userCursors.set(t,{...n,lastAction:"Added a comment"}),console.log("Comment saved:",s)}catch(s){console.error("Failed to save comment:",s)}}async loadExistingComments(){try{if(!this.currentRoom){console.log("Not in a room, skipping comment loading");return}const e=this.commentDB?await this.commentDB.getCommentsByUrlAndRoom(window.location.href,this.currentRoom):[];for(const t of e)await this.createHighlightFromComment(t),this.hasComments.add(t.highlightId);console.log(`Loaded ${e.length} existing comments for this page in room "${this.currentRoom}"`),this.webrtcChat&&e.length>0&&this.requestCommentStatus(e.map(t=>t.id))}catch(e){console.error("Failed to load existing comments:",e)}}requestCommentStatus(e){this.commentManager?this.commentManager.requestCommentStatus(e):console.warn("⚠️ Comment manager not initialized")}async onCommentStatusRequest(e,t){try{this.commentManager?await this.commentManager.handleCommentStatusRequest(e,t):console.warn("⚠️ Comment manager not initialized")}catch(o){console.error("Failed to handle comment status request:",o)}}async onCommentStatusResponse(e,t){try{this.commentManager?await this.commentManager.handleCommentStatusResponse(e,t):console.warn("⚠️ Comment manager not initialized")}catch(o){console.error("Failed to handle comment status response:",o)}}async onCommentAdd(e){try{this.commentManager&&this.commentManager.isInitialized?await this.commentManager.handleCommentAdd(e):(console.warn("⚠️ Comment manager not initialized - attempting to initialize and retry"),await this.initializeCommentSystem(),this.commentManager&&this.commentManager.isInitialized?await this.commentManager.handleCommentAdd(e):(console.warn("⚠️ Comment manager still not available - falling back to legacy system"),await this.handleCommentAddLegacy(e)))}catch(t){console.error("Failed to process incoming comment:",t)}}async handleCommentAddLegacy(e){try{if(e.url!==window.location.href||e.room!==this.currentRoom){console.log("Comment not for current page/room:",e.url,e.room,"vs",window.location.href,this.currentRoom);return}this.commentDB&&await this.commentDB.saveComment(e);const t=e.peerId;if(this.userCursors){const o=this.userCursors.get(t)||{};this.userCursors.set(t,{...o,lastAction:"Added a comment"})}await this.createHighlightFromComment(e),this.hasComments.add(e.highlightId),console.log("Received comment from peer (legacy):",e)}catch(t){console.error("Failed to process incoming comment (legacy):",t)}}async onCommentDelete(e){try{this.commentManager?await this.commentManager.handleCommentDelete(e):console.warn("⚠️ Comment manager not initialized")}catch(t){console.error("Failed to delete comment:",t)}}async resolveComment(e,t){try{this.commentManager?await this.commentManager.resolveComment(e):console.warn("⚠️ Comment manager not initialized")}catch(o){console.error("Failed to resolve comment:",o)}}async onCommentResolve(e,t,o=null,s=null,n=null){try{this.commentManager?await this.commentManager.handleCommentResolve(e,t,o,s,n):console.warn("⚠️ Comment manager not initialized")}catch(i){console.error("Failed to process incoming comment resolution:",i)}}onSystemMessage(e,t){this.addSystemMessage(e.message,e.url,e.author)}addSystemMessage(e,t,o=null){const s=this.shadowRoot.getElementById("chat-messages");if(!s)return;const n=document.createElement("div");n.className="chat-message system";let i="";if(t)try{const a=new URL(t),r=a.hostname,c=a.pathname,l=c&&c!=="/"?`View ${c}`:`View on ${r}`;i=`<a href="${t}" target="_blank" class="system-link" rel="noopener noreferrer">${l}</a>`}catch{i=`<a href="${t}" target="_blank" class="system-link" rel="noopener noreferrer">View Page</a>`}for(n.innerHTML=`
            <div class="system-message">
                <span class="system-icon">🔔</span>
                <span class="system-text">${this.escapeHtml(e)}</span>
                ${i}
            </div>
        `,s.appendChild(n),s.scrollTop=s.scrollHeight;s.children.length>50;)s.removeChild(s.firstChild)}onConnectionStatusChange(e){this.isConnected=e,this.updateUserStatsDisplay()}onUserCountChange(e){const t=this.shadowRoot.getElementById("user-count");t&&(t.textContent=e)}onMessage(e,t,o){if(t.type==="quiz_message")this.displayQuizMessage(t);else if(t.type==="ai_response")this.addChatMessage("MC",t.payload,o);else if(t.type==="quiz_submission")this.quizManager.handleQuizSubmissionFromPeer(t);else if(t.type==="quiz_lock")this.quizManager.handleQuizLockFromPeer(t);else if(t.type==="countdown_start")this.handleCountdownStart(t.payload);else if(this.addChatMessage(e,t,o),o){this.incrementUserAction("message");const s=this.webrtcChat.myClientId,n=this.userCursors.get(s)||{};this.userCursors.set(s,{...n,lastAction:"Sent a message",stars:this.userStats.totalStars,totalScore:this.userStats.totalScore,totalMessages:this.userStats.totalMessages})}else{const s=Object.keys(this.webrtcChat.peers).find(n=>this.webrtcChat.peers[n].username===e);if(s){const n=this.userCursors.get(s)||{};this.userCursors.set(s,{...n,lastAction:"Sent a message"})}}}async addChatMessage(e,t,o){console.log("🎮 addChatMessage called:",{username:e,message:t.substring(0,100)+"...",isOwn:o});const s=this.shadowRoot.getElementById("chat-messages");if(!s){console.error("🎮 Chat messages container not found!");return}console.log("🎮 Chat messages container found, adding message...");const n=document.createElement("div"),i=e==="MC"||e==="AI GO!";console.log("🎮 Processing message:",{isAIMessage:i,username:e});let a;if(i?(console.log("🎮 Parsing markdown for AI message..."),a=await this.parseMarkdownMessage(t),console.log("🎮 Markdown parsed successfully")):a=this.escapeHtml(t),i?n.className="chat-message ai-message":n.className=`chat-message ${o?"own":"other"}`,n.innerHTML=`
            <div class="username">${e}</div>
            <div class="message-text">${a}</div>
        `,s.appendChild(n),s.scrollTop=s.scrollHeight,i&&setTimeout(()=>{this.quizManager.setupQuizInteractions(n)},100),this.pruneOldMessages(),o&&this.userCursors){const r=this.webrtcChat.myClientId,c=this.userCursors.get(r)||{};this.userCursors.set(r,{...c,lastAction:"Sent a message"})}}addSystemMessage(e){const t=this.shadowRoot.getElementById("chat-messages");if(!t)return;const o=document.createElement("div");o.className="chat-message system",o.innerHTML=`<div class="message-text">${this.escapeHtml(e)}</div>`,t.appendChild(o),t.scrollTop=t.scrollHeight}escapeHtml(e){const t=document.createElement("div");return t.textContent=e,t.innerHTML}onInfoMessage(e,t){console.log(`[${t.toUpperCase()}] ${e}`),t==="success"&&e.includes("has connected")&&this.addSystemMessage(e)}onTypingIndicatorChange(e,t){const o=this.shadowRoot.getElementById("typing-indicator");o&&(e?(o.textContent=`${e} is typing...`,o.style.display="block"):o.style.display="none")}onUserJoined(e){console.log(`User ${e} joined`)}onUserLeft(e,t){console.log(`User ${t} left`),t&&t!=="...joining..."&&this.addSystemMessage(`${t} has left the room.`)}disconnect(){this.webrtcChat&&this.webrtcChat.disconnect(),this.clearRoomSession()}async displayQuizMessage(e){try{const{QuizDisplay:t}=await Promise.resolve().then(()=>V),o=this.shadowRoot.getElementById("chat-messages");if(!o)return;const s=t.createAIMessageElement(e.quiz);o.appendChild(s),o.scrollTop=o.scrollHeight,t.setupQuizInteraction(s,n=>{this.quizManager.handleQuizAnswer(n)}),console.log("Quiz displayed from AI:",e.quiz.question)}catch(t){console.error("Failed to display quiz message:",t),this.addChatMessage("AI GO!",`Quiz: ${e.quiz.question}`,!1)}}handleCountdownStart(e){console.log("🎮 Collaborative widget received countdown start event:",e),this.gamificationComponent&&this.gamificationComponent.handleCountdownStart?(console.log("🎮 Forwarding countdown to gamification component"),this.gamificationComponent.handleCountdownStart(e)):console.log("❌ Gamification component not available or missing handleCountdownStart method")}async handleChatMessage(e){try{const{ManualAIHandler:t}=await Promise.resolve().then(()=>L);t.isAITrigger(e)?(this.addChatMessage(this.webrtcChat.myUsername,e,!0),this.webrtcChat.broadcastMessage({type:"chat",payload:e}),await this.handleManualAIRequest(e)):this.webrtcChat.sendMessage(e)}catch(t){console.error("Failed to handle chat message:",t),this.webrtcChat.sendMessage(e)}}async handleManualAIRequest(e){try{const{ManualAIHandler:t}=await Promise.resolve().then(()=>L);this.addChatMessage("MC","Thinking...",!1);const o=this.webrtcChat.getCurrentPageData(),s=this.webrtcChat.quizRoomManager?{lastCollectedContent:this.webrtcChat.quizRoomManager.lastCollectedContent}:null,n=this.getRecentConversationHistory(),i=await t.handleManualAIRequest(e,o,s,n);await this.replaceLastMessage("MC",i),this.webrtcChat.broadcastMessage({type:"ai_response",payload:i,originalRequest:e,username:"MC"}),console.log("Manual AI request completed:",e)}catch(t){console.error("Manual AI request failed:",t),await this.replaceLastMessage("MC","Sorry, I encountered an error. Please try again.")}}async replaceLastMessage(e,t){const o=this.shadowRoot.getElementById("chat-messages");if(!o||o.children.length===0)return;const s=o.lastElementChild;if(s&&s.classList.contains("chat-message")){const n=s.querySelector(".message-text");if(n)if(e==="MC"||e==="AI GO!")try{const a=await this.parseMarkdownMessage(t);s.className="chat-message ai-message",n.innerHTML=a,setTimeout(()=>{this.quizManager.setupQuizInteractions(s)},100)}catch(a){console.error("Failed to parse markdown in replaceLastMessage:",a),n.textContent=t}else n.textContent=t;else if(e==="MC"||e==="AI GO!")try{const a=await this.parseMarkdownMessage(t);s.className="chat-message ai-message",s.innerHTML=`
                            <div class="username">${e}</div>
                            <div class="message-text">${a}</div>
                        `,setTimeout(()=>{this.quizManager.setupQuizInteractions(s)},100)}catch(a){console.error("Failed to parse markdown in replaceLastMessage:",a),s.innerHTML=`
                            <div class="username">${e}</div>
                            <div class="message-text">${this.escapeHtml(t)}</div>
                        `}else s.innerHTML=`
                        <div class="username">${e}</div>
                        <div class="message-text">${this.escapeHtml(t)}</div>
                    `}}showAIUsageHelp(){setTimeout(()=>{this.addSystemMessage("🎉 Welcome! Use @MC followed by your question to get help from our Master of Ceremonies! Example: @MC What is this page about?")},2e3)}async parseMarkdownMessage(e){try{console.log("🎮 parseMarkdownMessage called with message length:",e.length);const{MarkdownParser:t}=await Promise.resolve().then(()=>oe),o=await t.parseAll(e);return console.log("🎮 Markdown parsing completed, result length:",o.length),o}catch(t){return console.error("🎮 Failed to parse markdown:",t),this.escapeHtml(e)}}getRecentConversationHistory(){const e=this.shadowRoot.getElementById("chat-messages");return e?Array.from(e.children).slice(-6).map(s=>{const n=s.querySelector(".username")?.textContent||"Unknown",i=s.querySelector(".message-text")?.textContent||"";return{username:n,message:i}}).filter(s=>s.message.trim()!==""):[]}pruneOldMessages(){const e=this.shadowRoot.getElementById("chat-messages");if(!e)return;const t=Array.from(e.children);t.length>100&&t.slice(0,t.length-50).forEach(s=>e.removeChild(s))}setupThemeObserver(){const e=document.body,t=()=>{const s=e.classList.contains("dark-mode")||window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches;this.setTheme(s?"dark":"light")};new MutationObserver(s=>{for(const n of s)n.type==="attributes"&&n.attributeName==="class"&&t()}).observe(e,{attributes:!0}),window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change",t),t()}setTheme(e){const t=this.shadowRoot?.querySelector(".widget-container");t&&t.setAttribute("theme",e),this.options.theme=e}}class Y{static MAX_CHARS=6e3;static NAV_SELECTORS=["nav","header","footer",".nav",".navbar",".navigation",".menu",".sidebar",".breadcrumb",".pagination",'[role="navigation"]','[role="banner"]','[role="contentinfo"]'];static extractPageText(e=!0){const t=window.location.href,o=document.title;let s,n;if(e){const a=document.cloneNode(!0);this.NAV_SELECTORS.forEach(c=>{a.querySelectorAll(c).forEach(d=>d.remove())}),["script","style","noscript",".advertisement",".ads",".social-media",".share-buttons",".comments",".comment-section"].forEach(c=>{a.querySelectorAll(c).forEach(d=>d.remove())}),s=a.body.textContent||""}else s=document.body.textContent||"";s=this.cleanText(s),n=s.length;let i=!1;return s.length>this.MAX_CHARS&&(s=this.randomSample(s,this.MAX_CHARS),i=!0),{text:s,charCount:s.length,wordCount:this.getWordCount(s),isSampled:i,originalLength:n,url:t,title:o}}static cleanText(e){return e.replace(/\s+/g," ").replace(/\n\s*\n/g,`
`).trim()}static getWordCount(e){return!e||typeof e!="string"?0:e.split(/\s+/).filter(t=>t.length>0).length}static randomSample(e,t){if(e.length<=t)return e;const o=e.split(/[.!?]+/).filter(i=>i.trim().length>0);if(o.length===0)return this.randomWordSample(e,t);const s=this.shuffleArray([...o]);let n="";for(const i of s){const a=i.trim();if(n.length+a.length+1<=t)n+=(n?". ":"")+a;else break}return n+(n.endsWith(".")?"":".")}static randomWordSample(e,t){const o=e.split(/\s+/),s=this.shuffleArray([...o]);let n="";for(const i of s)if(n.length+i.length+1<=t)n+=(n?" ":"")+i;else break;return n}static shuffleArray(e){const t=[...e];for(let o=t.length-1;o>0;o--){const s=Math.floor(Math.random()*(o+1));[t[o],t[s]]=[t[s],t[o]]}return t}static getTextSummary(e){return`Page: ${e.title}
URL: ${e.url}
Characters: ${e.charCount}${e.isSampled?` (sampled from ${e.originalLength})`:""}
Preview: ${e.text.substring(0,100)}...`}static setupNavigationDetection(e,t={}){const{debounceMs:o=1e3,enableSPADetection:s=!0,enableHistoryDetection:n=!0,enableMutationDetection:i=!0}=t;let a=window.location.href,r=null;const c=()=>{const l=window.location.href;l!==a&&(console.log(`🔄 Page navigation detected: ${a} → ${l}`),a=l,r&&clearTimeout(r),r=setTimeout(()=>{console.log("📝 Extracting text from new page...");const d=this.extractPageText();e(d)},o))};if(n&&window.addEventListener("popstate",c),s){const l=history.pushState,d=history.replaceState;history.pushState=function(...h){l.apply(history,h),c()},history.replaceState=function(...h){d.apply(history,h),c()}}return i&&new MutationObserver(d=>{let h=!1;for(const m of d)if(m.type==="childList"&&m.addedNodes.length>0&&Array.from(m.addedNodes).some(f=>f.nodeType===Node.ELEMENT_NODE?f.tagName==="MAIN"||f.tagName==="ARTICLE"||f.classList?.contains("content")||f.classList?.contains("page-content")||f.id?.includes("content")||f.id?.includes("main"):!1)){h=!0;break}h&&c()}).observe(document.body,{childList:!0,subtree:!0}),window.addEventListener("hashchange",c),console.log("🎯 Navigation detection set up successfully"),()=>{r&&clearTimeout(r),window.removeEventListener("popstate",c),window.removeEventListener("hashchange",c),console.log("🧹 Navigation detection cleaned up")}}static hasPageChanged(e,t){const o=window.location.href,s=document.title;return o!==e||s!==t}}const q=Object.freeze(Object.defineProperty({__proto__:null,TextExtractor:Y},Symbol.toStringTag,{value:"Module"}));class y{static MAX_LINES_PER_USER=3;static MAX_CHARS_PER_LINE=200;static TOTAL_MAX_CHARS=1e3;static samplePageContent(e,t=this.MAX_LINES_PER_USER){if(!e||e.trim().length===0)return"No content available.";const s=this.cleanText(e).split(/[.!?]+/).map(a=>a.trim()).filter(a=>a.length>10&&a.length<this.MAX_CHARS_PER_LINE);return s.length===0?"Content too short to sample.":this.shuffleArray([...s]).slice(0,t).map(a=>a.trim()).join(". ")+"."}static cleanText(e){return e.replace(/\s+/g," ").replace(/\n\s*\n/g,`
`).trim()}static shuffleArray(e){const t=[...e];for(let o=t.length-1;o>0;o--){const s=Math.floor(Math.random()*(o+1));[t[o],t[s]]=[t[s],t[o]]}return t}static prepareContentForAI(e){const t=e.map(s=>({userId:s.userId,username:s.username,pageTitle:s.content.title,url:s.content.url,sampledText:this.samplePageContent(s.content.text,this.MAX_LINES_PER_USER)}));return t.reduce((s,n)=>s+n.sampledText.length,0)>this.TOTAL_MAX_CHARS&&t.forEach(s=>{s.sampledText.length>this.MAX_CHARS_PER_LINE&&(s.sampledText=s.sampledText.substring(0,this.MAX_CHARS_PER_LINE)+"...")}),{userCount:t.length,content:t,timestamp:Date.now(),totalCharacters:t.reduce((s,n)=>s+n.sampledText.length,0)}}static getContentSummary(e){return`Content from ${e.userCount} users (${e.totalCharacters} chars): ${e.content.map(t=>t.username).join(", ")}`}}class J{static injected=!1;static injectQuizStyles(){if(this.injected)return;const e=document.createElement("style");e.textContent=`
      /* Quiz Display Styles */
      .ai-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin: 8px 0;
        padding: 12px;
      }

      .ai-username {
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        margin-bottom: 8px;
        font-size: 14px;
      }

      .ai-username svg {
        width: 16px;
        height: 16px;
      }

      .quiz-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
      }

      .collaborative-quiz {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }

      .quiz-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        font-weight: 600;
        font-size: 16px;
      }

      .quiz-header svg {
        width: 20px;
        height: 20px;
      }

      .quiz-question {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 16px;
        font-size: 16px;
        line-height: 1.4;
      }

      .quiz-question svg {
        width: 16px;
        height: 16px;
        margin-top: 2px;
        flex-shrink: 0;
      }

      .quiz-question p {
        margin: 0;
        font-weight: 500;
      }

      .quiz-choices {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 16px;
      }

      .quiz-choice {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid transparent;
        border-radius: 8px;
        color: white;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 14px;
        text-align: left;
        width: 100%;
      }

      .quiz-choice:hover:not(:disabled) {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
      }

      .quiz-choice:disabled {
        cursor: not-allowed;
        opacity: 0.7;
      }

      .quiz-choice.selected {
        border-color: #4ade80;
        background: rgba(74, 222, 128, 0.2);
      }

      .quiz-choice.correct {
        border-color: #4ade80;
        background: rgba(74, 222, 128, 0.2);
      }

      .quiz-choice.incorrect {
        border-color: #ef4444;
        background: rgba(239, 68, 68, 0.2);
      }

      .choice-letter {
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        font-weight: 600;
        font-size: 12px;
        flex-shrink: 0;
      }

      .choice-text {
        flex: 1;
        line-height: 1.3;
      }

      .quiz-explanation {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
      }

      .explanation-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        font-weight: 600;
        font-size: 14px;
      }

      .explanation-header svg {
        width: 16px;
        height: 16px;
      }

      .explanations-content {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .explanation-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 8px;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.4;
      }

      .explanation-item.correct {
        background: rgba(74, 222, 128, 0.2);
      }

      .explanation-item.incorrect {
        background: rgba(239, 68, 68, 0.2);
      }

      .explanation-letter {
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        font-weight: 600;
        font-size: 11px;
        flex-shrink: 0;
      }

      .explanation-text {
        flex: 1;
      }

      .quiz-metadata {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 12px;
        font-size: 12px;
        opacity: 0.8;
      }

      .quiz-metadata svg {
        width: 14px;
        height: 14px;
      }

      .quiz-metadata span {
        display: flex;
        align-items: center;
        gap: 4px;
      }

      /* Leader selection styles */
      .leader-message {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
      }

      .leader-message svg {
        width: 14px;
        height: 14px;
      }

      /* Content collection styles */
      .collection-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
      }

      .collection-message svg {
        width: 14px;
        height: 14px;
      }

      /* Responsive design */
      @media (max-width: 480px) {
        .quiz-choice {
          padding: 10px;
          gap: 8px;
        }
        
        .choice-letter {
          width: 20px;
          height: 20px;
          font-size: 11px;
        }
        
        .quiz-question {
          font-size: 14px;
        }
        
        .quiz-metadata {
          flex-direction: column;
          align-items: flex-start;
          gap: 8px;
        }
      }

      /* Dark mode support */
      @media (prefers-color-scheme: dark) {
        .ai-message {
          background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 100%);
        }
        
        .quiz-container {
          background: rgba(0, 0, 0, 0.2);
        }
      }
    `,document.head.appendChild(e),this.injected=!0,console.log("Quiz styles injected successfully"),this.injectMarkdownStyles()}static injectMarkdownStyles(){if(document.querySelector("#markdown-styles"))return;const e=document.createElement("style");e.id="markdown-styles",e.textContent=`
      /* Markdown elements in chat messages */
      .chat-message .message-text h1,
      .chat-message .message-text h2,
      .chat-message .message-text h3 {
        margin: 8px 0 4px 0;
        font-weight: 600;
        line-height: 1.3;
      }

      .chat-message .message-text h1 {
        font-size: 18px;
        color: #2563eb;
      }

      .chat-message .message-text h2 {
        font-size: 16px;
        color: #1d4ed8;
      }

      .chat-message .message-text h3 {
        font-size: 14px;
        color: #1e40af;
      }

      .chat-message .message-text strong {
        font-weight: 600;
        color: #1f2937;
      }

      .chat-message .message-text em {
        font-style: italic;
        color: #374151;
      }

      .chat-message .message-text code {
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 13px;
        color: #dc2626;
      }

      .chat-message .message-text pre {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
        overflow-x: auto;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 13px;
        line-height: 1.4;
      }

      .chat-message .message-text pre code {
        background: none;
        border: none;
        padding: 0;
        color: #1f2937;
        font-size: inherit;
      }

      .chat-message .message-text a {
        color: #2563eb;
        text-decoration: underline;
        text-decoration-color: #93c5fd;
        transition: color 0.2s ease;
      }

      .chat-message .message-text a:hover {
        color: #1d4ed8;
        text-decoration-color: #60a5fa;
      }

      .chat-message .message-text ul {
        margin: 8px 0;
        padding-left: 20px;
      }

      .chat-message .message-text li {
        margin: 4px 0;
        line-height: 1.4;
      }

      .chat-message .message-text p {
        margin: 6px 0;
        line-height: 1.5;
      }

      .chat-message .message-text p:first-child {
        margin-top: 0;
      }

      .chat-message .message-text p:last-child {
        margin-bottom: 0;
      }

      /* Special styling for MC messages */
      .ai-message .message-text h1,
      .ai-message .message-text h2,
      .ai-message .message-text h3 {
        color: rgba(255, 255, 255, 0.9);
      }

      .ai-message .message-text strong {
        color: rgba(255, 255, 255, 0.95);
      }

      .ai-message .message-text em {
        color: rgba(255, 255, 255, 0.85);
      }

      .ai-message .message-text code {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
        color: rgba(255, 255, 255, 0.9);
      }

      .ai-message .message-text pre {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
      }

      .ai-message .message-text pre code {
        color: rgba(255, 255, 255, 0.9);
      }

      .ai-message .message-text a {
        color: rgba(255, 255, 255, 0.9);
        text-decoration-color: rgba(255, 255, 255, 0.6);
      }

      .ai-message .message-text a:hover {
        color: rgba(255, 255, 255, 1);
        text-decoration-color: rgba(255, 255, 255, 0.8);
      }

      /* Dark mode support */
      @media (prefers-color-scheme: dark) {
        .chat-message .message-text h1 {
          color: #60a5fa;
        }

        .chat-message .message-text h2 {
          color: #3b82f6;
        }

        .chat-message .message-text h3 {
          color: #2563eb;
        }

        .chat-message .message-text strong {
          color: #f9fafb;
        }

        .chat-message .message-text em {
          color: #e5e7eb;
        }

        .chat-message .message-text code {
          background-color: #374151;
          border-color: #4b5563;
          color: #fca5a5;
        }

        .chat-message .message-text pre {
          background-color: #1f2937;
          border-color: #374151;
        }

        .chat-message .message-text pre code {
          color: #f9fafb;
        }

        .chat-message .message-text a {
          color: #60a5fa;
          text-decoration-color: #3b82f6;
        }

        .chat-message .message-text a:hover {
          color: #93c5fd;
          text-decoration-color: #60a5fa;
        }
      }

      /* Responsive design */
      @media (max-width: 480px) {
        .chat-message .message-text h1 {
          font-size: 16px;
        }

        .chat-message .message-text h2 {
          font-size: 14px;
        }

        .chat-message .message-text h3 {
          font-size: 13px;
        }

        .chat-message .message-text pre {
          padding: 8px;
          font-size: 12px;
        }

        .chat-message .message-text code {
          font-size: 12px;
          padding: 1px 4px;
        }
      }
    `,document.head.appendChild(e),console.log("Markdown styles injected successfully")}}class S{static AI_USERNAME="AI GO!";static AI_USER_ID="ai-go";static createQuizHTML(e){return`
      <div class="collaborative-quiz" data-quiz-id="${e.id||Date.now()}" data-correct-answer="${e.correctAnswer}">
        <div class="quiz-header">
          ${this.getQuestionIcon()}
          <span class="quiz-title">Collaborative Quiz</span>
        </div>
        
        <div class="quiz-question">
          ${this.getQuestionIcon()}
          <p>${e.question}</p>
        </div>
        
        <div class="quiz-choices">
          ${Object.entries(e.choices).map(([t,o])=>`
            <button class="quiz-choice" data-choice="${t}" data-quiz-id="${e.id||Date.now()}">
              <span class="choice-letter">${t}</span>
              <span class="choice-text">${o}</span>
            </button>
          `).join("")}
        </div>
        
        <div class="quiz-explanation" style="display: none;">
          <div class="explanation-header">
            ${this.getExplanationIcon()}
            <span>Explanations</span>
          </div>
          <div class="explanations-content">
            ${Object.entries(e.explanations).map(([t,o])=>`
              <div class="explanation-item ${t===e.correctAnswer?"correct":"incorrect"}">
                <span class="explanation-letter">${t}</span>
                <span class="explanation-text">${o}</span>
              </div>
            `).join("")}
          </div>
        </div>
        
        <div class="quiz-metadata">
          ${this.getDifficultyIcon()}
          <span>Difficulty: ${e.difficulty}</span>
          ${this.getUsersIcon()}
          <span>${e.metadata.userCount} contributors</span>
        </div>
      </div>
    `}static createAIMessageElement(e){J.injectQuizStyles();const t=document.createElement("div");t.className="chat-message ai-message";const o=this.createQuizHTML(e);return t.innerHTML=`
      <div class="username ai-username">
        ${this.getAIIcon()}
        ${this.AI_USERNAME}
      </div>
      <div class="quiz-container">
        ${o}
      </div>
    `,t}static handleQuizAnswer(e,t){const o=e.dataset.choice,s=e.dataset.quizId,n=e.closest(".collaborative-quiz"),i=n.dataset.correctAnswer;n.querySelectorAll(".quiz-choice").forEach(c=>c.disabled=!0),e.classList.add("selected"),o===i?e.classList.add("correct"):(e.classList.add("incorrect"),n.querySelector(`[data-choice="${i}"]`).classList.add("correct"));const r=n.querySelector(".quiz-explanation");r.style.display="block",t&&t({quizId:s,choice:o,correctAnswer:i,isCorrect:o===i,timestamp:Date.now()})}static setupQuizInteraction(e,t){e.addEventListener("click",o=>{const s=o.target.closest(".quiz-choice");s&&!s.disabled&&this.handleQuizAnswer(s,t)})}static getAIIcon(){return`<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M9 12l2 2 4-4"/>
      <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/>
      <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"/>
      <path d="M12 3c0 1-1 3-3 3s-3-2-3-3 1-3 3-3 3 2 3 3"/>
      <path d="M12 21c0-1 1-3 3-3s3 2 3 3-1 3-3 3-3-2-3-3"/>
    </svg>`}static getQuestionIcon(){return`<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/>
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>`}static getExplanationIcon(){return`<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14,2 14,8 20,8"/>
      <line x1="16" y1="13" x2="8" y2="13"/>
      <line x1="16" y1="17" x2="8" y2="17"/>
      <polyline points="10,9 9,9 8,9"/>
    </svg>`}static getDifficultyIcon(){return`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/>
      <path d="M12 6v6l4 2"/>
    </svg>`}static getUsersIcon(){return`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
      <circle cx="12" cy="7" r="4"/>
    </svg>`}static getLeaderIcon(){return`<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
    </svg>`}static getCollectionIcon(){return`<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      <path d="M13 8H7"/>
      <path d="M17 12H7"/>
    </svg>`}}const V=Object.freeze(Object.defineProperty({__proto__:null,QuizDisplay:S},Symbol.toStringTag,{value:"Module"}));class X{constructor(e,t){this.roomName=e,this.webrtcChat=t,this.members=new Map,this.currentLeader=null,this.lastAICall=0,this.collectionInProgress=!1,this.collectionTimeout=null,this.periodicCollectionInterval=null,this.baseInterval=5*60*1e3,this.minWaitTime=30*1e3,this.maxWaitTime=2*60*1e3,this.lastCollectedContent=null}addUser(e,t){this.members.set(e,{userId:e,username:t,lastSeen:Date.now(),isLeader:!1}),console.log(`User ${t} added to room ${this.roomName}. Total members: ${this.members.size}`),this.members.size===1&&this.selectNewLeader()}removeUser(e){const t=this.members.get(e);t&&(this.members.delete(e),console.log(`User ${t.username} removed from room ${this.roomName}. Total members: ${this.members.size}`),this.currentLeader===e&&this.selectNewLeader())}selectNewLeader(){if(this.members.size===0){this.currentLeader=null;return}const e=Array.from(this.members.keys()),t=e[Math.floor(Math.random()*e.length)];this.members.forEach((s,n)=>{s.isLeader=n===t}),this.currentLeader=t;const o=this.members.get(t)?.username;console.log(`New leader selected for room ${this.roomName}: ${o} (${t})`),this.broadcastToRoom({type:"leader_selected",leaderId:t,leaderName:o,timestamp:Date.now()})}isCurrentUserLeader(){return this.currentLeader===this.webrtcChat.myClientId}async startContentCollection(){if(!(!this.isCurrentUserLeader()||this.collectionInProgress)){console.log(`Leader ${this.webrtcChat.myUsername} collecting content for room ${this.roomName} (AI disabled)`);try{const e=await this.collectContentFromAll(),t=y.prepareContentForAI(e);console.log(`Content collected: ${y.getContentSummary(t)}`),this.lastCollectedContent=t,console.log("Content collection complete. AI calling disabled - use @AI to trigger manual AI.")}catch(e){console.error("Content collection failed:",e)}}}async collectContentFromAll(){if(this.collectionInProgress)return[];this.collectionInProgress=!0;const e=crypto.randomUUID(),t=[];console.log("🎮 Starting content collection with requestId:",e),console.log("🎮 Current room members:",this.members.size);const o=s=>{s.type==="content_response"&&s.requestId===e&&(console.log("🎮 Received content response from:",s.username),console.log("🎮 Response content preview:",s.content?.text?.substring(0,100)+"..."),t.push(s))};return this.currentResponseHandler=o,this.broadcastToRoom({type:"collect_content",requestId:e,leaderId:this.currentLeader,timestamp:Date.now()}),console.log("🎮 Broadcasted content collection request"),new Promise(s=>{this.collectionTimeout=setTimeout(()=>{this.collectionInProgress=!1,this.currentResponseHandler=null,console.log(`🎮 Content collection timeout. Received ${t.length} responses out of ${this.members.size-1} expected`),console.log("🎮 Responses received:",t.map(i=>({username:i.username,url:i.content?.url,textLength:i.content?.text?.length}))),s(t)},5e3);const n=()=>{const i=this.members.size-1;t.length>=i?(clearTimeout(this.collectionTimeout),this.collectionInProgress=!1,this.currentResponseHandler=null,console.log(`🎮 Content collection completed. Received ${t.length} responses`),console.log("🎮 Final responses:",t.map(a=>({username:a.username,url:a.content?.url,textLength:a.content?.text?.length}))),s(t)):setTimeout(n,100)};n()})}async startImmediateContentCollection(){if(console.log("🎮 Starting immediate content collection for Kapoot game"),this.collectionInProgress)return console.log("🎮 Content collection already in progress"),this.lastCollectedContent||null;this.collectionInProgress=!0;const e=crypto.randomUUID(),t=[],o=s=>{s.type==="content_response"&&s.requestId===e&&(t.push(s),console.log(`🎮 Received content response from ${s.username}: ${s.content.url}`))};return this.currentResponseHandler=o,this.webrtcChat.onMessage=(s,n,i)=>{o(n),this.webrtcChat.originalOnMessage&&this.webrtcChat.originalOnMessage(s,n,i)},this.broadcastToRoom({type:"collect_content",requestId:e,leaderId:this.currentLeader,timestamp:Date.now(),purpose:"kapoot_game_start"}),console.log("🎮 Broadcasted content collection request to all users"),new Promise(s=>{this.collectionTimeout=setTimeout(()=>{this.collectionInProgress=!1,this.currentResponseHandler=null,console.log(`🎮 Content collection timeout. Received ${t.length} responses`);const i=this.processCollectedContent(t);this.lastCollectedContent=i,s(i)},1e4);const n=()=>{if(t.length>=this.members.size-1){clearTimeout(this.collectionTimeout),this.collectionInProgress=!1,this.currentResponseHandler=null;const i=this.processCollectedContent(t);this.lastCollectedContent=i,console.log("🎮 All users responded to content collection"),s(i)}else setTimeout(n,500)};setTimeout(n,500)})}processCollectedContent(e){const t={userCount:e.length,content:e.map(o=>({userId:o.userId,username:o.username,pageTitle:o.content.title,url:o.content.url,sampledText:this.samplePageContent(o.content.text),timestamp:o.timestamp})),timestamp:Date.now(),totalCharacters:e.reduce((o,s)=>o+(s.content.text?.length||0),0)};return console.log("🎮 Processed content data:",{userCount:t.userCount,totalCharacters:t.totalCharacters,urls:t.content.map(o=>o.url)}),t}samplePageContent(e,t=3){return!e||e==="No content available"?"No content available":e.split(`
`).filter(n=>n.trim().length>0).slice(0,t).map(n=>n.substring(0,200).trim()).join(" ")}shouldTriggerAI(e){const o=Date.now()-this.lastAICall;return e.userCount>=this.members.size||o>this.baseInterval}async generateAndDistributeQuiz(e){try{console.log(`Generating quiz from content: ${y.getContentSummary(e)}`);const t=this.minWaitTime+Math.random()*(this.maxWaitTime-this.minWaitTime);console.log(`Waiting ${Math.round(t/1e3)}s before AI call...`),await new Promise(n=>setTimeout(n,t));const o=await x.generateQuiz(e),s={type:"quiz_message",from:S.AI_USERNAME,userId:S.AI_USER_ID,timestamp:Date.now(),quiz:o,metadata:{roomName:this.roomName,userCount:e.userCount,contentSources:e.content.map(n=>n.pageTitle)}};this.broadcastToRoom(s),this.lastAICall=Date.now(),this.selectNewLeader(),console.log(`Quiz generated and distributed for room ${this.roomName}`)}catch(t){console.error("Quiz generation failed:",t)}}broadcastToRoom(e){this.webrtcChat&&this.webrtcChat.broadcastMessage&&this.webrtcChat.broadcastMessage(e)}handleContentCollectionRequest(e){if(e.type==="collect_content"&&e.leaderId!==this.webrtcChat.myClientId){console.log("🎮 Received content collection request for Kapoot game");const t=this.webrtcChat.currentPageData||{text:"No content available",url:window.location.href,title:document.title,timestamp:Date.now()},o={type:"content_response",requestId:e.requestId,userId:this.webrtcChat.myClientId,username:this.webrtcChat.myUsername,content:t,timestamp:Date.now()};console.log("🎮 Sending content response:",{url:t.url,title:t.title,textLength:t.text?.length||0,textPreview:t.text?.substring(0,100)+"...",fullContent:t}),this.webrtcChat.broadcastMessage(o)}}handleContentResponse(e){e.type==="content_response"&&(console.log(`🎮 Received content response from ${e.username}: ${e.content.url}`),console.log("🎮 Content response details:",{username:e.username,url:e.content.url,title:e.content.title,textLength:e.content.text?.length||0,textPreview:e.content.text?.substring(0,100)+"...",fullContent:e.content}),this.currentResponseHandler&&this.currentResponseHandler(e))}startPeriodicCollection(){this.periodicCollectionInterval=setInterval(()=>{this.isCurrentUserLeader()&&this.startContentCollection()},2*60*1e3)}stopPeriodicCollection(){this.periodicCollectionInterval&&(clearInterval(this.periodicCollectionInterval),this.periodicCollectionInterval=null),this.collectionTimeout&&(clearTimeout(this.collectionTimeout),this.collectionTimeout=null),this.collectionInProgress=!1,this.currentResponseHandler=null,console.log("🧹 AI room manager cleanup completed")}getRoomStatus(){return{roomName:this.roomName,memberCount:this.members.size,currentLeader:this.currentLeader,leaderName:this.members.get(this.currentLeader)?.username,lastAICall:this.lastAICall,collectionInProgress:this.collectionInProgress}}}const Z=Object.freeze(Object.defineProperty({__proto__:null,QuizRoomManager:X},Symbol.toStringTag,{value:"Module"}));class ee{static AI_USERNAME="MC";static AI_USER_ID="master-of-ceremonies";static async handleManualAIRequest(e,t,o,s=[]){try{const n=this.extractQuestion(e);if(!n)return"Please provide a question after @MC. For example: @MC What is this page about?";const i=y.samplePageContent(t.text,5),a=this.buildManualAIPrompt(n,i,t,o,s);return await x.callAIProxy(a)}catch(n){return console.error("Manual AI request failed:",n),"Sorry, I encountered an error processing your request. Please try again."}}static extractQuestion(e){const t=e.match(/^@mc\s+(.+)$/i);return t?t[1].trim():null}static buildManualAIPrompt(e,t,o,s,n=[]){const i=o.title||"this page",a=o.url||"current page";let r="";if(s&&s.lastCollectedContent){const l=s.lastCollectedContent;r=`

Additional context from the collaborative room (${l.userCount} users):
${l.content.map(d=>`${d.username} (${d.pageTitle}): ${d.sampledText}`).join(`

`)}`}let c="";return n.length>0&&(c=`

Recent conversation:
`+n.map(l=>`${l.username}: ${l.message}`).join(`
`)),{messages:[{role:"system",content:`You are the Master of Ceremonies (MC) for a collaborative learning app! You're here to create an engaging and helpful atmosphere while being informative.

Your Personality:
- Helpful assistant who answers questions about content accurately
- Creates engaging, social interactions
- Uses occasional emojis for personality (limit to 1-2 per response)
- Can be playful while staying informative and clear

Guidelines:
- Be enthusiastic but not overwhelming
- Use emojis sparingly (1-2 per response maximum)
- Answer questions about page content accurately
- Create a collaborative atmosphere
- Keep responses conversational and helpful
- If you don't have enough information, say so clearly
- Focus on being informative rather than just entertaining
- Reference previous conversation when relevant

Special Capabilities:
- You can create interactive quizzes using ONLY these tags: <quiz> and </quiz>
- When asked for a quiz, wrap your quiz JSON in <quiz>JSON</quiz> tags
- Quiz format: {"question": "...", "choices": {"A": "...", "B": "...", "C": "...", "D": "..."}, "correctAnswer": "A", "explanations": {"A": "...", "B": "...", "C": "...", "D": "..."}}
- You can also display JSON data using <json>JSON</json> tags
- NEVER use QUIZTOKEN, ENDQUIZTOKEN, or any other format
- ONLY use <quiz> and </quiz> with angle brackets
- Example: <quiz>{"question": "What is 2+2?", "choices": {"A": "3", "B": "4", "C": "5", "D": "6"}, "correctAnswer": "B", "explanations": {"A": "Incorrect", "B": "Correct!", "C": "Incorrect", "D": "Incorrect"}}</quiz>
- Always use proper JSON formatting within the tags`},{role:"user",content:`Question: ${e}

Page Content from "${i}" (${a}):
${t}${r}${c}

Please provide a helpful response.`}],model:"gpt-4o-mini",temperature:.7,max_tokens:500}}static isAITrigger(e){return/^@mc\s+/i.test(e)}static getTriggerPattern(){return"@MC or @mc followed by your question"}static getExampleUsage(){return"Examples: @MC What is this page about? @mc Can you summarize the key points?"}}const L=Object.freeze(Object.defineProperty({__proto__:null,ManualAIHandler:ee},Symbol.toStringTag,{value:"Module"}));class te{static parse(e){if(!e||typeof e!="string")return"";let t=e;return t=this.parseHeaders(t),t=this.parseBold(t),t=this.parseItalic(t),t=this.parseCodeBlocks(t),t=this.parseInlineCode(t),t=this.parseImages(t),t=this.parseLinks(t),t=this.parseTables(t),t=this.parseLists(t),t=this.parseLineBreaks(t),t}static escapeHtml(e){const t=document.createElement("div");return t.textContent=e,t.innerHTML}static escapeHtmlPreservingMarkers(e){const t=["___QUIZ_TOKEN___","___END_QUIZ_TOKEN___","___JSON_TOKEN___","___END_JSON_TOKEN___"],o=["___SAFE_QUIZ_START___","___SAFE_QUIZ_END___","___SAFE_JSON_START___","___SAFE_JSON_END___"];let s=e;return t.forEach((n,i)=>{s=s.replace(new RegExp(n.replace(/[.*+?^${}()|[\]\\]/g,"\\$&"),"g"),o[i])}),s=this.escapeHtml(s),o.forEach((n,i)=>{s=s.replace(new RegExp(n.replace(/[.*+?^${}()|[\]\\]/g,"\\$&"),"g"),t[i])}),s}static parseHeaders(e){return e.replace(/^### (.*$)/gm,"<h3>$1</h3>").replace(/^## (.*$)/gm,"<h2>$1</h2>").replace(/^# (.*$)/gm,"<h1>$1</h1>")}static parseBold(e){return e.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>").replace(/__(.*?)__/g,"<strong>$1</strong>")}static parseItalic(e){return e.replace(/\*(.*?)\*/g,"<em>$1</em>").replace(/_(.*?)_/g,"<em>$1</em>")}static parseCodeBlocks(e){return e.replace(/```([\s\S]*?)```/g,"<pre><code>$1</code></pre>")}static parseInlineCode(e){return e.replace(/`([^`]+)`/g,"<code>$1</code>")}static parseLinks(e){return e.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')}static parseImages(e){return e.replace(/!\[([^\]]*)\]\(([^)]+)\)/g,'<img src="$2" alt="$1" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;" />')}static parseTables(e){const t=/^(\|.*\|)\n(\|[\s\-\|]*\|)\n((?:\|.*\|\n?)*)/gm;return e.replace(t,(o,s,n,i)=>{const r=s.split("|").slice(1,-1).map(d=>d.trim()).map(d=>`<th>${d}</th>`).join(""),l=i.trim().split(`
`).filter(d=>d.trim()).map(d=>`<tr>${d.split("|").slice(1,-1).map(u=>u.trim()).map(u=>`<td>${u}</td>`).join("")}</tr>`).join("");return`<table class="markdown-table">
<thead>
<tr>${r}</tr>
</thead>
<tbody>
${l}
</tbody>
</table>`})}static parseLists(e){return e=e.replace(/^[\s]*[-*] (.*$)/gm,"<li>$1</li>"),e=e.replace(/(<li>.*<\/li>)(\s*<li>.*<\/li>)*/g,t=>"<ul>"+t+"</ul>"),e}static parseLineBreaks(e){return e=e.replace(/\n\n/g,"</p><p>"),e=e.replace(/\n/g,"<br>"),e="<p>"+e+"</p>",e=e.replace(/<p><\/p>/g,""),e=e.replace(/<p>\s*<\/p>/g,""),e}static parseEmojis(e){const t={":smile:":"😊",":laugh:":"😂",":heart:":"❤️",":thumbsup:":"👍",":fire:":"🔥",":star:":"⭐",":rocket:":"🚀",":party:":"🎉",":confetti:":"🎊",":tada:":"🎉",":sparkles:":"✨",":bulb:":"💡",":check:":"✅",":cross:":"❌",":warning:":"⚠️",":question:":"❓",":exclamation:":"❗"};let o=e;for(const[s,n]of Object.entries(t))o=o.replace(new RegExp(s,"g"),n);return o}static async parseAll(e){if(!e||typeof e!="string")return"";let t=this.parseEmojis(e);return t=this.parse(t),t=await this.parseComponents(t),t}static async parseComponents(e){try{const{MessageComponentRegistry:t}=await Promise.resolve().then(()=>ae);return await t.parseAndRender(e)}catch(t){return console.error("Failed to parse components:",t),e}}static hasMarkdown(e){return e?[/^#{1,6}\s/,/\*\*.*?\*\*/,/\*.*?\*/,/`.*?`/,/```[\s\S]*?```/,/\[.*?\]\(.*?\)/,/^[\s]*[-*]\s/,/^\|.*\|$/m].some(o=>o.test(e)):!1}}const oe=Object.freeze(Object.defineProperty({__proto__:null,MarkdownParser:te},Symbol.toStringTag,{value:"Module"}));class se{static async render(e){if(!this.validateQuizData(e))return'<div class="component-error">Invalid quiz data format</div>';const t=e.quizId||`quiz-${Date.now()}-${Math.random().toString(36).substr(2,9)}`;return`
      <div class="message-quiz-component" data-quiz-id="${t}" data-correct-answer="${e.correctAnswer}">
        <div class="quiz-header">
          <svg class="quiz-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 12l2 2 4-4"/>
            <path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/>
            <path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"/>
            <path d="M12 3c0 1-1 3-3 3s-3-2-3-3 1-3 3-3 3 2 3 3"/>
            <path d="M12 21c0-1 1-3 3-3s3 2 3 3-1 3-3 3-3-2-3-3"/>
          </svg>
          <span class="quiz-title">Interactive Quiz</span>
        </div>
        
        <div class="quiz-question">
          <p>${e.question}</p>
        </div>
        
        <div class="quiz-choices">
          ${Object.entries(e.choices).map(([o,s])=>`
            <label class="quiz-choice" data-choice="${o}" data-quiz-id="${t}">
              <input type="radio" name="quiz-${t}" value="${o}" class="quiz-radio">
              <span class="choice-letter">${o}</span>
              <span class="choice-text">${s}</span>
            </label>
          `).join("")}
        </div>
        
        <div class="quiz-submit-container">
          <button class="quiz-submit-btn" data-quiz-id="${t}" disabled>
            <svg class="submit-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 19l7-7 3 3-7 7-3-3z"/>
              <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
              <path d="M2 2l7.586 7.586"/>
              <circle cx="11" cy="11" r="2"/>
            </svg>
            <span class="submit-text">Submit Answer</span>
          </button>
        </div>
        
        <div class="quiz-explanation" style="display: none;">
          <div class="explanation-header">
            <svg class="explanation-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
              <path d="M12 17h.01"/>
            </svg>
            <span>Explanations</span>
          </div>
          <div class="explanations-content">
            ${Object.entries(e.explanations).map(([o,s])=>`
              <div class="explanation-item ${o===e.correctAnswer?"correct":"incorrect"}">
                <span class="explanation-letter">${o}</span>
                <span class="explanation-text">${s}</span>
              </div>
            `).join("")}
          </div>
        </div>
        
        <div class="quiz-result" style="display: none;">
          <svg class="result-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
            <line x1="9" y1="9" x2="9.01" y2="9"/>
            <line x1="15" y1="9" x2="15.01" y2="9"/>
          </svg>
          <span class="result-text"></span>
        </div>
      </div>
    `}static validateQuizData(e){if(!e||typeof e!="object")return!1;const t=["question","choices","correctAnswer","explanations"];for(const o of t)if(!e[o])return!1;return!(!e.choices||typeof e.choices!="object"||Object.keys(e.choices).length<2||!e.choices[e.correctAnswer]||!e.explanations||typeof e.explanations!="object")}}class ne{static async render(e){try{const t=JSON.stringify(e,null,2);return`
        <div class="message-json-component">
          <div class="json-header">
            <div class="json-icon">📋</div>
            <span class="json-title">Data</span>
          </div>
          <div class="json-content">
            <pre><code>${this.escapeHtml(t)}</code></pre>
          </div>
        </div>
      `}catch(t){return console.error("Failed to render JSON component:",t),'<div class="component-error">Invalid JSON data</div>'}}static escapeHtml(e){const t=document.createElement("div");return t.textContent=e,t.innerHTML}}class k{constructor(){this.activeQuizzes=new Map,this.currentUserTeam=null}static async render(e){return new k().renderKapootQuiz(e)}renderKapootQuiz(e){return e.timerExpired||e.explanationsRevealed?this.renderResultsPhase(e):this.renderActivePhase(e)}renderActivePhase(e){const t=e.specialFeatures?.showProgressBar?this.createProgressBar(e.quizNumber,e.totalQuizzes):"";return`
            <div class="quiz-kapoot-active" data-quiz-id="${e.quizId}">
                <script type="application/json" class="quiz-data">${JSON.stringify(e)}<\/script>
                ${t}
                <div class="quiz-kapoot-header">
                    <span class="quiz-kapoot-title">🎮 Kapoot Team Quiz Challenge!</span>
                    <span class="quiz-kapoot-timer" id="timer-${e.quizId}">${Math.ceil(e.remainingTime/1e3)}s ⏰</span>
                </div>
                
                <div class="quiz-kapoot-question-info">
                    <span class="quiz-kapoot-question-number">Quiz ${e.quizNumber||1} of ${e.totalQuizzes||3}</span>
                    <span class="quiz-kapoot-difficulty">${e.difficulty||"Intermediate"}</span>
                </div>
                
                <div class="quiz-kapoot-question">${e.question}</div>
                
                <div class="quiz-kapoot-choices">
                    ${Object.entries(e.choices).map(([o,s])=>`
                        <button class="quiz-kapoot-choice" data-choice="${o}" data-quiz-id="${e.quizId}">
                            <span class="quiz-choice-letter">${o}</span>
                            <span class="quiz-choice-text">${s}</span>
                        </button>
                    `).join("")}
                </div>
                
                <div class="quiz-kapoot-team-status">
                    <strong>Teams:</strong> ${Object.keys(e.teamResponses).join(", ")}<br>
                    <strong>Status:</strong> ${this.getTeamStatusText(e.teamResponses)}
                </div>
                
                <div class="quiz-kapoot-sources">
                    <strong>📚 Source Pages:</strong><br>
                    ${e.sourceUrls.map((o,s)=>`${s+1}. <a href="${o.url}" target="_blank">${o.title}</a>${o.username&&o.username!=="System"?` - ${o.username}`:""}`).join("<br>")}
                </div>
                
                <div class="quiz-kapoot-note">
                    ❌ Explanations hidden until timer expires
                </div>
            </div>
        `}renderResultsPhase(e){return`
            <div class="quiz-kapoot-results" data-quiz-id="${e.quizId}">
                <div class="quiz-kapoot-header">
                    <span class="quiz-kapoot-title">🎮 Kapoot Quiz Results!</span>
                </div>
                
                <div class="quiz-kapoot-question">${e.question}</div>
                
                <div class="quiz-kapoot-choices">
                    ${Object.entries(e.choices).map(([t,o])=>`
                        <div class="quiz-kapoot-choice-result ${t===e.correctAnswer?"correct":"incorrect"}">
                            <span class="quiz-choice-letter">${t}</span>
                            <span class="quiz-choice-text">${o}</span>
                            <span class="quiz-choice-indicator">${t===e.correctAnswer?"✅":"❌"}</span>
                        </div>
                    `).join("")}
                </div>
                
                <div class="quiz-kapoot-correct-answer">
                    <strong>Correct Answer:</strong> ${e.correctAnswer}
                </div>
                
                <div class="quiz-kapoot-team-results">
                    <strong>🏆 Team Results:</strong><br>
                    ${Object.entries(e.teamResponses).map(([t,o])=>`${t}: ${o.answered?o.answer:"No answer"} - ${o.score} pts`).join("<br>")}
                </div>
                
                <div class="quiz-kapoot-explanations">
                    <strong>💡 Explanations:</strong><br>
                    ${Object.entries(e.explanations).map(([t,o])=>`${t}) ${o}`).join("<br>")}
                </div>
                
                <div class="quiz-kapoot-sources">
                    <strong>📚 Source Pages:</strong><br>
                    ${e.sourceUrls.map((t,o)=>`${o+1}. <a href="${t.url}" target="_blank">${t.title}</a>${t.username&&t.username!=="System"?` - ${t.username}`:""}`).join("<br>")}
                </div>
            </div>
        `}setupKapootQuizInteractions(e){const t=document.querySelector(`[data-quiz-id="${e.quizId}"]`);if(!t)return;t.querySelectorAll(".quiz-kapoot-choice").forEach(s=>{s.addEventListener("click",n=>{const i=n.currentTarget.dataset.choice;this.handleTeamAnswer(e,i)})})}handleTeamAnswer(e,t){const o=this.findCurrentUserTeam(e);if(!o){console.warn("🎮 No team found for current user");return}if(e.teamResponses[o].answered){console.log("🎮 Team already answered:",o);return}const s=Date.now()-e.startTime,n=this.calculateTeamScore(s,e.basePoints,t===e.correctAnswer);e.teamResponses[o]={answered:!0,answer:t,responseTime:s,score:n},console.log("🎮 Team answer received:",{team:o,choice:t,responseTime:s,score:n}),this.updateKapootQuizDisplay(e),Object.values(e.teamResponses).every(a=>a.answered)&&(console.log("🎮 All teams answered, revealing explanations early"),this.revealExplanationsEarly(e)),this.broadcastTeamResponse(e,o,t,s,n)}findCurrentUserTeam(e){if(e.currentUserTeam&&e.teamResponses[e.currentUserTeam])return this.currentUserTeam=e.currentUserTeam,e.currentUserTeam;if(this.currentUserTeam&&e.teamResponses[this.currentUserTeam])return this.currentUserTeam;const t=Object.keys(e.teamResponses).filter(o=>!e.teamResponses[o].answered);return t.length>0?(this.currentUserTeam=t[0],this.currentUserTeam):null}calculateTeamScore(e,t,o){if(!o)return 0;const s=3e4-e,n=Math.max(0,s/3e4),i=Math.floor(t*n);return console.log("🎮 Score calculation:",{responseTime:e,basePoints:t,timeRemaining:s,scoreMultiplier:n,finalScore:i}),i}startKapootTimer(e){e.startTime=Date.now(),e.remainingTime=e.timeLimit||3e4;const t=setInterval(()=>{e.remainingTime=Math.max(0,e.remainingTime-1e3);const o=document.getElementById(`timer-${e.quizId}`);if(o){o.textContent=`${Math.ceil(e.remainingTime/1e3)}s ⏰`;const s=this.getUrgencyLevel(e.remainingTime);o.className=`quiz-kapoot-timer ${s}`}e.remainingTime<=0&&(clearInterval(t),this.handleTimerExpiry(e))},1e3);this.activeQuizzes.set(e.quizId,{quiz:e,timerInterval:t}),console.log("🎮 Kapoot timer started for quiz:",e.quizId)}handleTimerExpiry(e){console.log("🎮 Kapoot timer expired for quiz:",e.quizId),e.timerExpired=!0,e.explanationsRevealed=!0,this.updateKapootQuizDisplay(e),this.broadcastTimerExpiry(e)}revealExplanationsEarly(e){console.log("🎮 Revealing explanations early for quiz:",e.quizId),e.timerExpired=!0,e.explanationsRevealed=!0;const t=this.activeQuizzes.get(e.quizId);t&&t.timerInterval&&clearInterval(t.timerInterval),this.updateKapootQuizDisplay(e),this.broadcastExplanationReveal(e)}updateKapootQuizDisplay(e){const t=document.querySelector(`[data-quiz-id="${e.quizId}"]`);if(!t){console.warn("🎮 Quiz element not found for update:",e.quizId);return}const o=this.renderKapootQuiz(e);t.outerHTML=o,this.setupKapootQuizInteractions(e),console.log("🎮 Updated Kapoot quiz display:",e.quizId)}getTeamStatusText(e){return Object.entries(e).map(([t,o])=>`${t}: ${o.answered?"✓ Answered":"⏳ Thinking"}`).join(" | ")}getUrgencyLevel(e){const t=e/3e4;return t>.5?"safe":t>.2?"warning":"critical"}broadcastTeamResponse(e,t,o,s,n){const i=new CustomEvent("kapootTeamResponse",{detail:{quizId:e.quizId,team:t,answer:o,responseTime:s,score:n}});document.dispatchEvent(i)}broadcastTimerExpiry(e){const t=new CustomEvent("kapootTimerExpiry",{detail:{quizId:e.quizId,quiz:e}});document.dispatchEvent(t)}broadcastExplanationReveal(e){const t=new CustomEvent("kapootExplanationReveal",{detail:{quizId:e.quizId,quiz:e}});document.dispatchEvent(t)}cleanupQuiz(e){const t=this.activeQuizzes.get(e);t&&t.timerInterval&&clearInterval(t.timerInterval),this.activeQuizzes.delete(e),console.log("🎮 Cleaned up Kapoot quiz:",e)}handleIncomingTeamResponse(e){const{quizId:t,team:o,answer:s,responseTime:n,score:i}=e,a=this.activeQuizzes.get(t);if(!a){console.warn("🎮 Received response for unknown quiz:",t);return}const r=a.quiz;r.teamResponses[o]={answered:!0,answer:s,responseTime:n,score:i},console.log("🎮 Received team response:",{quizId:t,team:o,answer:s,score:i}),this.updateKapootQuizDisplay(r),Object.values(r.teamResponses).every(l=>l.answered)&&this.revealExplanationsEarly(r)}handleIncomingTimerExpiry(e){const{quizId:t,quiz:o}=e,s=this.activeQuizzes.get(t);s&&Object.assign(s.quiz,o),this.updateKapootQuizDisplay(o),console.log("🎮 Received timer expiry for quiz:",t)}createProgressBar(e,t){const o=e/t*100;return`
            <div class="quiz-progress-container">
                <div class="quiz-progress-text">Quiz ${e} of ${t}</div>
                <div class="quiz-progress-bar">
                    <div class="quiz-progress-fill" style="width: ${o}%"></div>
                </div>
            </div>
        `}}class ie{static components=new Map([["quiz",se],["json",ne],["quiz-kapoot",k]]);static async parseAndRender(e){let t=e;return t=await this.parseComponent(t,"quiz"),t=await this.parseComponent(t,"quiz-kapoot"),t=await this.parseComponent(t,"json"),t}static async parseComponent(e,t){const o=this.components.get(t);if(!o)return e;const s=`<${t}>`,n=`</${t}>`;let i=e,a=i.indexOf(s);for(;a!==-1;){const r=i.indexOf(n,a);if(r===-1)break;const c=i.substring(a+s.length,r).trim();try{let l=c.replace(/<br\s*\/?>/gi,"").replace(/<[^>]*>/g,"").replace(/[\r\n\t]/g,"").trim();console.log(`Parsing ${t} JSON:`,l);const d=JSON.parse(l);if(t==="quiz"&&!d.quizId){const m=this.hashString(l);d.quizId=`quiz-${m}`}const h=await o.render(d);i=i.substring(0,a)+h+i.substring(r+n.length)}catch(l){console.error(`Failed to parse ${t} component:`,l),console.error("JSON content was:",c),i=i.substring(0,a)+`<div class="component-error">Invalid ${t} format: ${l.message}</div>`+i.substring(r+n.length)}a=i.indexOf(s,a)}return i}static registerComponent(e,t){this.components.set(e,t)}static hashString(e){let t=0;if(e.length===0)return t.toString();for(let o=0;o<e.length;o++){const s=e.charCodeAt(o);t=(t<<5)-t+s,t=t&t}return Math.abs(t).toString(36)}}const ae=Object.freeze(Object.defineProperty({__proto__:null,MessageComponentRegistry:ie},Symbol.toStringTag,{value:"Module"}));p.CollaborativeWidget=A,p.GamificationComponent=E,p.QuizManager=C,p.StatEffects=$,p.WebRTCChat=w,p.WidgetResizer=R,p.default=A,p.gamificationStyles=M,p.htmlContent=z,p.styles=T,Object.defineProperties(p,{__esModule:{value:!0},[Symbol.toStringTag]:{value:"Module"}})});

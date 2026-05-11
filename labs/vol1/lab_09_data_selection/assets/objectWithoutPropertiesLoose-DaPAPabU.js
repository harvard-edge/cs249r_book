function f(n,t){if(n==null)return{};var r={};for(var i in n)if({}.hasOwnProperty.call(n,i)){if(t.indexOf(i)!==-1)continue;r[i]=n[i]}return r}export{f as t};

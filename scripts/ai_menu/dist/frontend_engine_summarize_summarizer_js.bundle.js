/*
 * ATTENTION: The "eval" devtool has been used (maybe by default in mode: "development").
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(self["webpackChunkinjectchat"] = self["webpackChunkinjectchat"] || []).push([["frontend_engine_summarize_summarizer_js"],{

/***/ "./frontend_engine/summarize/summarizer.js":
/*!*************************************************!*\
  !*** ./frontend_engine/summarize/summarizer.js ***!
  \*************************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   summarizeTextWithSingleton: () => (/* binding */ summarizeTextWithSingleton)\n/* harmony export */ });\n/* harmony import */ var _xenova_transformers__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @xenova/transformers */ \"./node_modules/@xenova/transformers/src/transformers.js\");\n\r\n\r\n\r\n// Skip initial check for local models, since we are not loading any local models.\r\n_xenova_transformers__WEBPACK_IMPORTED_MODULE_0__.env.allowLocalModels = false;\r\n\r\nclass SummarizerSingleton {\r\n    static instance = null;\r\n\r\n    static async getInstance() {\r\n        if (!this.instance) {\r\n            // Assuming 'pipeline' is a function that initializes models similar to Hugging Face's Transformers\r\n            this.instance = await (0,_xenova_transformers__WEBPACK_IMPORTED_MODULE_0__.pipeline)('summarization', 'Xenova/distilbart-cnn-6-6');\r\n        }\r\n        return this.instance;\r\n    }\r\n}\r\n\r\nconst summarizeTextWithSingleton = async (text) => {\r\n    console.log(\"i am text in summarizer\", text)\r\n    try {\r\n        const summarizer = await SummarizerSingleton.getInstance();\r\n        console.log(summarizer)\r\n        const output = await summarizer(text, {\r\n            max_new_tokens: 300, // Adjust based on how detailed you want the summary to be\r\n        });\r\n        return output; // Assuming the output is the summarized text\r\n    } catch (error) {\r\n        console.error(\"Error within summarizeTextWithSingleton:\", error);\r\n        throw error;\r\n    }\r\n};\r\n\r\n// // Example usage\r\n// (async () => {\r\n//     const text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, ' +\r\n//       'and the tallest structure in Paris...'; // Example text\r\n//     try {\r\n//         const summary = await summarizeTextWithSingleton(text);\r\n//         console.log(\"Summary:\", summary);\r\n//     } catch (error) {\r\n//         console.error(\"Error during summarization process:\", error);\r\n//     }\r\n// })();\r\n\n\n//# sourceURL=webpack://injectchat/./frontend_engine/summarize/summarizer.js?");

/***/ }),

/***/ "?2ca1":
/*!**********************************!*\
  !*** onnxruntime-node (ignored) ***!
  \**********************************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/onnxruntime-node_(ignored)?");

/***/ }),

/***/ "?0a40":
/*!********************!*\
  !*** fs (ignored) ***!
  \********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/fs_(ignored)?");

/***/ }),

/***/ "?61c2":
/*!**********************!*\
  !*** path (ignored) ***!
  \**********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/path_(ignored)?");

/***/ }),

/***/ "?0740":
/*!***********************!*\
  !*** sharp (ignored) ***!
  \***********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/sharp_(ignored)?");

/***/ }),

/***/ "?66bb":
/*!****************************!*\
  !*** stream/web (ignored) ***!
  \****************************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/stream/web_(ignored)?");

/***/ }),

/***/ "?0a9a":
/*!********************!*\
  !*** fs (ignored) ***!
  \********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/fs_(ignored)?");

/***/ }),

/***/ "?73ea":
/*!**********************!*\
  !*** path (ignored) ***!
  \**********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/path_(ignored)?");

/***/ }),

/***/ "?845f":
/*!*********************!*\
  !*** url (ignored) ***!
  \*********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/url_(ignored)?");

/***/ })

}]);
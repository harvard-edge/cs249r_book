import {cloneElementById} from '../utils/utils'
// import {copy_download} from '../../components/settings/copy_download.js'
/**
 * Generates a message element based on the type given.
 *
 * @param {Element} shadowEle - The shadow element to clone from
 * @param {string} type - The type of message to generate ('ai' or other)
 * @return {Element} The cloned message element
 */
export function get_message_element(shadowEle, type){
    let clone;
    if (type === 'ai'){
    clone = cloneElementById(shadowEle, "ai-message", '',   'message-container')
    // copy_download(clone)
    }
    else{

    clone = cloneElementById(shadowEle, "human-message", '',   'message-container')
    }

    
    return clone;
}






/**
 * Generates reference buttons using the provided shadow element, clone, and links.
 *
 * @param {Element} shadowEle - The shadow DOM element to work with
 * @param {Element} clone - The clone element to work with
 * @param {Array<string>} links - The array of links to use for creating buttons
 */
export function get_reference_buttons (shadowEle, clone, links){
    // const container = cloneElementById(shadowEle, "reference-btn-container", '',)
// const container =  clone.querySelector('#reference-btn-container')
const container = cloneElementById(shadowEle, "reference-btn-container", '')

 links.forEach((link, i) => {

    const a = cloneElementById(shadowEle, "reference-btn", '')
    a.setAttribute('href', link)
    a.textContent = i + 1
    container.appendChild(a);
  
 })  
//  clone.appendChild(container);

return container

}
function formatDate(date) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return date.toLocaleDateString('en-US', options);
}

export function createCitation(style='') {
    const title = document.title;
    const url = window.location.href;
    const accessDate = formatDate(new Date());
    const author = document.querySelector('meta[name="author"]') ? document.querySelector('meta[name="author"]').getAttribute('content') : "No author specified";
    const publicationDate = document.querySelector('meta[name="publication-date"]') ? document.querySelector('meta[name="publication-date"]').getAttribute('content') : "No publication date specified";

    let citation = "";

    // switch (style) {
    //     case 'APA':
            citation = `${author} (${publicationDate}). ${title}. Retrieved ${accessDate}, from ${url}`;
        //     break;
        // case 'Chicago':
        //     citation = `${author}. "${title}." Last modified ${publicationDate}. Accessed ${accessDate}. ${url}.`;
        //     break;
        // case 'Harvard':
        //     citation = `${author} (${publicationDate}) '${title}', Available at: ${url} (Accessed: ${accessDate}).`;
        //     break;
        // default:
        //     citation = "Citation style not recognized.";
        //     break;
    // }

    return citation;
}

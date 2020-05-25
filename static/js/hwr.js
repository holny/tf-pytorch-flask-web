$(function () {


});

function checkIsNullOrEmpty(value) {
    //正则表达式用于判斷字符串是否全部由空格或换行符组成
    var reg = /^\s*$/;
    //返回值为true表示不是空字符串
    return (value != null && value != undefined && !reg.test(value))
}

function sweetAlertCancel() {


}

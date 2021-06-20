const DrawWithKey = 90; // z
const MovePointWithKey = 88; // x
const DeleteTailWithKey = 46; // x

const DragBallSize = 5

const BG_WIDTH = 700
const BG_HEIGHT = 650

// Clickable代码不用去深究他，主要是实现了形状的事件机制，参考其他成熟的UI框架写的，除非出现重大bug，不然不用去看他的代码。
// 以下两个函数为坐标系转换函数，将画布的坐标转换为mat矩阵的下标
function tranToGrid(x, y) {//转到矩阵下标
    let tmpX = x, tmpY = BG_HEIGHT - y
    tmpX -= (bgobj.width - bgobj.scaled_w) / 2
    tmpY -= (bgobj.height - bgobj.scaled_h) / 2
    tmpX += DragBallSize / 2
    tmpY -= DragBallSize / 2
    tmpX = Math.ceil(tmpX / bgobj.scaled_w * bgobj.im_w)
    tmpY = Math.floor(tmpY / bgobj.scaled_h * bgobj.im_h)
    return [tmpX, tmpY]
}

function gridToPos(x, y) {//下标到画布坐标
    let tmpX = x, tmpY = y
    tmpX = tmpX * bgobj.scaled_w / bgobj.im_w
    tmpY = tmpY * bgobj.scaled_h / bgobj.im_h
    tmpX -= DragBallSize / 2
    tmpY += DragBallSize / 2
    tmpX += (bgobj.width - bgobj.scaled_w) / 2
    tmpY += (bgobj.height - bgobj.scaled_h) / 2
    return [tmpX, BG_HEIGHT - tmpY]
}

function onHover() { // 小圆点：鼠标悬浮事件
    this.color = "#AAAAFF";
    this.textColor = "#FFFFFF";
    this.visiable = true;
}

function onOutside() { //小圆点：鼠标离开事件
    this.color = "#EEEEEE";
    this.textColor = "#000000";
    this.visiable = false;
}

function onPress() { //小圆点：鼠标按下事件
    this.visiable = true;
    // if (!keyIsDown(MovePointWithKey)) return;
    if (!this.isPressing) {
        this.isPressing = true;
        this.prevX = this.innerX;
        this.prevY = this.innerY;
    }
    this.stroke = "#ff0000"; // 网格对齐
    let grid_pos = tranToGrid(mouseX - this.prevX, mouseY - this.prevY)
    grid_pos = gridToPos(grid_pos[0], grid_pos[1])
    this.locate(grid_pos[0], grid_pos[1])
}

function onRelease() { // 小圆点：鼠标释放事件
    this.isPressing = false
}

var pointsList = []
var vertexList = [] //所有的点储存在这个列表中
let bgobj = null; //背景对象
let n = 0;

function setup() {
    createCanvas(BG_WIDTH, BG_HEIGHT);
    bgobj = new Clickable();
    bgobj.color = "#FFFFFF";
    bgobj.resize(BG_WIDTH, BG_HEIGHT);
    bgobj.locate(BG_WIDTH / 2, BG_HEIGHT / 2);
    bgobj.cornerRadius = 0
    bgobj.strokeWeight = 0
    bgobj.im_scale = 1

    bgobj.onPress = () => {//给背景绑定按下事件
        //z
        if (!this.isPressing && keyIsDown(DrawWithKey)) {//如果此时按下了Z键，那么就在鼠标位置加一个点
            this.isPressing = true;
            let click = new Clickable();
            click.cornerRadius = DragBallSize;
            click.resize(DragBallSize * 2, DragBallSize * 2)
            click.locate(mouseX - DragBallSize, mouseY - DragBallSize);

            click.onHover = onHover
            click.onOutside = onOutside
            click.onPress = onPress
            click.onRelease = onRelease
            window.vertexList[window.vertexList.length] = click
        }

    }
    bgobj.onRelease = () => {
        this.isPressing = false
    }

    // 可以通过一下方式创建点集，比如从后端传过来的结点数据，就可以通过以下方式进行创建

    // let click = new Clickable();
    // click.cornerRadius = DragBallSize;
    // click.resize(DragBallSize * 2, DragBallSize * 2)
    // click.locate(mouseX - DragBallSize, mouseY - DragBallSize);
    //
    // click.onHover = onHover
    // click.onOutside = onOutside
    // click.onPress = onPress
    // click.onRelease = onRelease
    //
    // vertexList[vertexList.length] = click

}

function ConnectLine(src, des) { // 画线
    stroke("#f50e63");
    let stroke_weight = bgobj.scaled_h / bgobj.im_h/3
    strokeWeight(stroke_weight);
    line(src.x + stroke_weight / 2, src.y,
        des.x + stroke_weight / 2, des.y);
}

function draw() { // 一直在刷新，这个函数会一直由p5.js持续回调执行
    background(100);
    bgobj.draw(); //先画背景
    for (let i = 0; i < window.vertexList.length; i++) {//再画线
        ConnectLine(window.vertexList[i], window.vertexList[(i + 1) % window.vertexList.length])
    }
    for (let i = 0; i < window.vertexList.length; i++) {//最后画点
        window.vertexList[i].draw()
    }
}
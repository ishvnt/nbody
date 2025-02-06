#ifndef POINTS_H
#define POINTS_H

class Point
{
    public:
        double x;
        double y;
        double vx;
        double vy;
        Point() {this->x = 0.0f; this->y = 0.0f; this->vx = 0.0f; this->vx = 0.0f;};
        Point(double x, double y) {this->x = x; this->y = y; this->vx = 0.0f; this->vy = 0.0f;};
        Point(double x, double y, double vx, double vy) {this->x = x; this->y = y; this->vx = vx; this->vy = vy;};
};

#endif
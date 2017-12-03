/*
SQLyog Community v12.4.3 (64 bit)
MySQL - 5.7.19-log : Database - dengueprediction
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`dengueprediction` /*!40100 DEFAULT CHARACTER SET utf8 */;

USE `dengueprediction`;

/*Table structure for table `cities` */

DROP TABLE IF EXISTS `cities`;

CREATE TABLE `cities` (
  `cityname` varchar(50) NOT NULL,
  `lng` float NOT NULL,
  `lat` float NOT NULL,
  PRIMARY KEY (`cityname`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `cities` */

insert  into `cities`(`cityname`,`lng`,`lat`) values 
('iq',-73.2538,-3.74912),
('sj',-66.1057,18.4663);

/*Table structure for table `predictiondata` */

DROP TABLE IF EXISTS `predictiondata`;

CREATE TABLE `predictiondata` (
  `id` int(50) NOT NULL AUTO_INCREMENT,
  `city` varchar(10) NOT NULL,
  `year` int(4) NOT NULL,
  `week` int(2) NOT NULL,
  `cases` float NOT NULL,
  `deleteflag` tinyint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=833 DEFAULT CHARSET=utf8;

/*Data for the table `predictiondata` */

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
